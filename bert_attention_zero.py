#!/usr/bin/python
# author kingbone

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.models.bert.modeling_bert import (
    BertConfig, BertIntermediate, BertOutput, BertPreTrainedModel, BertModel
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"


class SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, config, n_labels):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.n_labels = n_labels
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # batch, seq_len, n_heads, head_size
        return x.permute(0, 2, 1, 3)  # batch, n_heads, seq_len, head_size

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                output_attentions=False,
                labels_embedding=None,
                ):
        device = hidden_states.device
        batch_size = hidden_states.size(0)
        # hidden_states(batch_size,seq_length,hidden_size) labels_embedding(batch_size,labels_num,hidden_size)
        hidden_states = torch.cat((labels_embedding, hidden_states), dim=1)
        temp = torch.zeros(batch_size, 1, 1, 148).to(device)

        mixed_query_layer = self.query(hidden_states[:, :self.n_labels, :])
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # (batch, n_heads, labels_num, seq_length+labels_num)
        attention_scores = attention_scores / math.sqrt(self.all_head_size)
        if attention_mask is not None:
            attention_mask = torch.cat((temp, attention_mask), dim=-1)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # (batch, n_heads, labels_num, hidden_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # (batch, labels_num, hidden_size)

        return context_layer


class FinalLayer(nn.Module):
    def __init__(self, config, n_labels):
        super().__init__()
        self.n_labels = n_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention = Attention(config, n_labels)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            labels_embedding=None,
    ):
        self_attn = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            labels_embedding=labels_embedding
        )  # (batch, n_labels, hidden_size)
        intermediate_output = self.intermediate(self_attn)
        context_output = self.output(intermediate_output, self_attn)  # (batch, n_labels, hidden_size)
        batch_size = context_output.size(0)
        context_output = context_output.view(batch_size, -1)  # (batch, n_labels * hidden_size)
        return context_output


class Linear_Classifier(nn.Module):
    def __init__(self, config, labels_num):
        super().__init__()
        self.out_mesh_dstrbtn = nn.Linear(config.hidden_size * labels_num, labels_num)
        nn.init.xavier_uniform_(self.out_mesh_dstrbtn.weight)

    def forward(self, context_vectors):
        output_dstrbtn = self.out_mesh_dstrbtn(context_vectors)  # (batch, n_labels)
        output_dstrbtn = output_dstrbtn.unsqueeze(-1)
        return output_dstrbtn


class BertForClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 148
        self.config = config

        self.bert = BertModel(config)
        self.attn_layer = FinalLayer(config, self.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = Linear_Classifier(config, self.num_labels)
        self.max_pool = nn.MaxPool1d(4)

        self.init_weights()

    def forward(
            self,
            input_ids=None,  # 输入的id,模型会帮你把id转成embedding
            attention_mask=None,  # attention里的mask
            token_type_ids=None,  # [CLS]A[SEP]B[SEP]
            position_ids=None,  # 位置id
            head_mask=None,  # 哪个head需要被mask掉
            inputs_embeds=None,  # 可以选择不输入id,直接输入embedding
            labels_embedding=None,
    ):
        if labels_embedding is not None:
            batch_size = input_ids.size(0)
            labels_embedding = labels_embedding.expand(batch_size, -1, -1)
            encoder_outputs1 = self.bert(
                input_ids=input_ids[:, :500],
                attention_mask=attention_mask[:, :500],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            encoder_outputs2 = self.bert(
                input_ids=input_ids[:, 500:1000],
                attention_mask=attention_mask[:, 500:1000],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            encoder_outputs3 = self.bert(
                input_ids=input_ids[:, 1000:1500],
                attention_mask=attention_mask[:, 1000:1500],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            encoder_outputs4 = self.bert(
                input_ids=input_ids[:, 1500:2000],
                attention_mask=attention_mask[:, 1500:2000],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # batch, 1, 1, seq_len
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            attn_outputs1 = F.elu(self.attn_layer(encoder_outputs1[0],
                                                  attention_mask=extended_attention_mask[:, :, :, :500],
                                                  labels_embedding=labels_embedding))
            attn_outputs2 = F.elu(self.attn_layer(encoder_outputs2[0],
                                                  attention_mask=extended_attention_mask[:, :, :, 500:1000],
                                                  labels_embedding=labels_embedding))
            attn_outputs3 = F.elu(self.attn_layer(encoder_outputs3[0],
                                                  attention_mask=extended_attention_mask[:, :, :, 1000:1500],
                                                  labels_embedding=labels_embedding))
            attn_outputs4 = F.elu(self.attn_layer(encoder_outputs4[0],
                                                  attention_mask=extended_attention_mask[:, :, :, 1500:2000],
                                                  labels_embedding=labels_embedding))

            attn_outputs1 = self.dropout(attn_outputs1)
            attn_outputs2 = self.dropout(attn_outputs2)
            attn_outputs3 = self.dropout(attn_outputs3)
            attn_outputs4 = self.dropout(attn_outputs4)

            logits1 = self.classifier(attn_outputs1)
            logits2 = self.classifier(attn_outputs2)
            logits3 = self.classifier(attn_outputs3)
            logits4 = self.classifier(attn_outputs4)

            logits = torch.cat((logits1, logits2, logits3, logits4), 2)
            logits = self.max_pool(logits).squeeze(-1)

            outputs = (logits,)

            return outputs  # logits, pooled_output, sequence_output
        else:
            encoder_outputs = self.bert(
                input_ids=input_ids[:, :500],
                attention_mask=attention_mask[:, :500],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
            return encoder_outputs[1]
