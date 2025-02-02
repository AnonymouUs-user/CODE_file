import torch
import torch.nn as nn
import math
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.activations import ACT2FN


class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads)
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.output = BertOutput(hidden_size, intermediate_size)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor, output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output
        )
        outputs = (layer_output, outputs)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads)
        self.output = BertSelfOutput(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_2(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        new_x_shape = (batch_size, height, width, self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)  # 转换为 (batch_size, height, width, num_heads, head_size)
        x = x.permute(0, 3, 1, 2, 4)
        x = x.reshape(batch_size, self.num_attention_heads, -1,
                      self.attention_head_size)  # (batch_size, num_heads, seq_len, head_size)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        # mixed_query_layer.shape torch.Size([12842, 4, 14951])
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # key_layer.shape  torch.Size([12842, 1, 4, 14951])
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # value_layer.shape torch.Size([12842, 1, 4, 14951])
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # query_layer.shape torch.Size([12842, 1, 4, 14951])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores.shape torch.Size([12842, 1, 4, 4])
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        # context_layer.shape torch.Size([12842, 4, 14951])
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states