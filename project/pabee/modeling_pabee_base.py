import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm

from transformers import PreTrainedModel
from transformers.modeling_utils import PreTrainedModel

import time

from lazy_layers.lazy_module_list import LazyModuleList

class BaseEncoderWithPabee(nn.Module):
    def __init__(self, config, layer_cls, lazy=False):
        nn.Module.__init__(self)

        # hack for distilbert
        # TODO: check if other classes have other names
        if not hasattr(config, 'num_hidden_layers'):
            setattr(config, 'num_hidden_layers', config.n_layers)

        if lazy:
            self.layers = LazyModuleList(
                [(layer_cls, (config,), {}) for _ in range(config.num_hidden_layers)])
        else:
            self.layer = nn.ModuleList([layer_cls(config) for _ in range(config.num_hidden_layers)])

    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None):
        layer_outputs = self.layer[current_layer](hidden_states, attention_mask, head_mask[current_layer])
        hidden_states = layer_outputs[0]
        return hidden_states


class BasePabeeModel(PreTrainedModel):
    def __init__(self, config, layer_cls, lazy=False, encoder_varname=None):
        encoder = BaseEncoderWithPabee(config, layer_cls, lazy=lazy)
        # hack for distilbert
        if encoder_varname is not None:
            setattr(self, encoder_varname, encoder)
            self.encoder = getattr(self, encoder_varname)
        else:
            self.encoder = encoder

        self.init_weights()
        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0
        self.runtimes = float("Inf")

        self.regression_threshold = 0

    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold

    def set_patience(self, patience):
        self.patience = patience

    def set_runtimes(self, runtimes):
        self.runtimes = runtimes

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f"*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up = {1 - avg_inf_layers / self.config.num_hidden_layers:.2f} ***"
        print(message)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_dropout=None,
        output_layers=None,
        regression=False,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = embedding_output

        if self.training:
            res = []
            for i in range(self.config.num_hidden_layers):
                encoder_outputs = self.encoder.adaptive_forward(
                    encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                )

                pooled_output = self.pooler(encoder_outputs)
                logits = output_layers[i](output_dropout(pooled_output))
                res.append(logits)
        elif self.patience == 0:  # Use all layers for inference
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
            )
            pooled_output = self.pooler(encoder_outputs[0])
            res = [output_layers[self.config.num_hidden_layers - 1](pooled_output)]
        else:
            init_time = time.time()
            patient_counter = 0
            patient_result = None
            calculated_layer_num = 0
            for i in range(self.config.num_hidden_layers):
                calculated_layer_num += 1
                encoder_outputs = self.encoder.adaptive_forward(
                    encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                )

                pooled_output = self.pooler(encoder_outputs)
                logits = output_layers[i](pooled_output)
                if regression:
                    labels = logits.detach()
                    if patient_result is not None:
                        patient_labels = patient_result.detach()
                    if (patient_result is not None) and torch.abs(patient_result - labels) < self.regression_threshold:
                        patient_counter += 1
                    else:
                        patient_counter = 0
                else:
                    labels = logits.detach().argmax(dim=1)
                    if patient_result is not None:
                        patient_labels = patient_result.detach().argmax(dim=1)
                    if (patient_result is not None) and torch.all(labels.eq(patient_labels)):
                        patient_counter += 1
                    else:
                        patient_counter = 0

                patient_result = logits
                if patient_counter == self.patience:
                    break
                if self.runtime_threshold > time.time() - init_time:
                    break
            res = [patient_result]
            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1

        return res

