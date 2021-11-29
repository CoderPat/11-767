# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, The HuggingFace Inc. team and Microsoft Corporation.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model with Patience-based Early Exit. """


import logging
import time 
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.distilbert.modeling_distilbert import (
    DISTILBERT_INPUTS_DOCSTRING,
    DISTILBERT_START_DOCSTRING,
    TransformerBlock,
    Embeddings,
    DistilBertModel,
    DistilBertPreTrainedModel,
)

from transformers.models.bert.modeling_bert import BertPooler

from .modeling_pabee_base import BasePabeeModel


logger = logging.getLogger(__name__)



@add_start_docstrings(
    "The bare DistilBert Model transformer with PABEE outputting raw hidden-states without any specific head on top.",
    DISTILBERT_START_DOCSTRING,
)
class DistilBertModelWithPabee(BasePabeeModel, DistilBertModel):
    """
    #TODO: Add PABEE prefix docstring
    """
    def __init__(self, config, lazy=False):
        DistilBertPreTrainedModel.__init__(self, config)
        self.embeddings = Embeddings(config)
        # Add explicit pooler to make stuff easier
        # TODO: confirm this works fine
        self.pooler = BertPooler(config)

        BasePabeeModel.__init__(self, config, TransformerBlock, lazy=lazy, encoder_varname="transformer", simple_embedding=True)


@add_start_docstrings(
    """Bert Model transformer with PABEE and a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    DISTILBERT_START_DOCSTRING,
)
class DistilBertForSequenceClassificationWithPabee(DistilBertPreTrainedModel):
    def __init__(self, config, lazy=False):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModelWithPabee(config, lazy=lazy)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.classifiers = nn.ModuleList(
            [nn.Linear(config.dim, self.config.num_labels) for _ in range(config.num_hidden_layers)]
        )

        self.init_weights()

    @add_start_docstrings_to_model_forward(DISTILBERT_START_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                Classification (or regression if config.num_labels==1) loss.
            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.

        Examples::

            from transformers import BertTokenizer, BertForSequenceClassification
            from pabee import BertForSequenceClassificationWithPabee
            from torch import nn
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForSequenceClassificationWithPabee.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, labels=labels)

            loss, logits = outputs[:2]

        """
        logits = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_dropout=self.dropout,
            output_layers=self.classifiers,
            regression=self.num_labels == 1,
        )

        outputs = (logits[-1],)

        if labels is not None:
            total_loss = None
            total_weights = 0
            for ix, logits_item in enumerate(logits):
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits_item.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits_item.view(-1, self.num_labels), labels.view(-1))
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss * (len(logits) - ix)
                total_weights += ix + 1
            outputs = (total_loss / total_weights,) + outputs

        return outputs

    def save_splitted_checkpoint(self, splitted_checkpoint):
        configuration = self.config
        if not os.path.exists(splitted_checkpoint):
            os.makedirs(splitted_checkpoint)

        state_dict = self.state_dict()
        state_dict = self.distilbert.save_splitted_layers(splitted_checkpoint, state_dict=state_dict)
    
        torch.save(state_dict, os.path.join(splitted_checkpoint, "model.pt"))
        torch.save(configuration, os.path.join(splitted_checkpoint, "config.pt"))


    def load_splitted_checkpoint(self, splitted_checkpoint):
        self.load_state_dict(torch.load(os.path.join(splitted_checkpoint, "model.pt")), strict=False)
        if not hasattr(self.config, "filenames"):
            setattr(self.config, "filenames", [])
        for i in range(self.config.num_hidden_layers):
            self.config.filenames.append(os.path.join(splitted_checkpoint, f"layer_{i}.pt"))
