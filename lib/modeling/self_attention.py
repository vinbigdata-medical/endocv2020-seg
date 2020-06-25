import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAModule(nn.Module):
    def __init__(self, channels, reduction, act_layer, num_attention_heads=1):
        super(SAModule, self).__init__()
        self.all_head_size = channels // reduction
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(self.all_head_size / self.num_attention_heads)

        self.query = nn.Conv2d(
            channels, self.all_head_size, kernel_size=1, padding=0,
        )
        self.key = nn.Conv2d(
            channels, self.all_head_size, kernel_size=1, padding=0,
        )
        self.value = nn.Conv2d(
            channels, self.all_head_size, kernel_size=1, padding=0,
        )
        self.output = nn.Conv2d(
            self.all_head_size, channels, kernel_size=1, padding=0,
        )
        self.gamma = nn.Parameter(torch.FloatTensor([0.]))
        if act_layer is not None:
            self.act = act_layer(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        query_layer = self.query(x)
        key_layer = self.key(x)
        value_layer = self.value(x)

        # single-headed attention
        attention_scores = torch.matmul(
            key_layer.view(x.size(0), self.all_head_size, -1).transpose(-1, -2),
            query_layer.view(x.size(0), self.all_head_size, -1))
        attention_scores = attention_scores / math.sqrt(self.all_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(
            value_layer.view(x.size(0), self.all_head_size, -1),
            attention_probs)
        context_layer = context_layer.view(
            x.size(0), self.all_head_size, x.size(-2), x.size(-1))
        output_layer = self.output(context_layer)
        x = x + output_layer * self.gamma
        if self.act is not None:
            x = self.act(x)
        return x
