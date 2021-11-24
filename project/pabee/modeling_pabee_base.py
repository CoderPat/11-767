import torch
import torch.nn as nn

from transformers import Transformer, TransformerBlock, PreTrainedModel
from transformers.modeling_utils import PreTrainedModel

from lazy_layers.lazy_module_list import LazyModuleList


class EncoderWithPabee(Transformer):
    def __init__(self, config, lazy):
        super().__init__()
        self.n_layers = config.n_layers
        if lazy:
            self.layers = LazyModuleList(
                [(TransformerBlock, (config,), {}) for _ in range(config.n_layers)])
        else:
            self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None):
        layer_outputs = self.layer[current_layer](hidden_states, attention_mask, head_mask[current_layer])
        hidden_states = layer_outputs[0]
        return hidden_states


class PabeeModel(PreTrainedModel):
    def __init__(self, config, lazy=False):
        super().__init__(config)

        self.encoder = EncoderWithPabee(config, lazy=lazy)

        self.init_weights()
        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0

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
        r"""
        Return:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction (classification)
                objective during pre-training.
                This output is usually *not* a good summary
                of the semantic content of the input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

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
            res = [patient_result]
            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1

        return res

