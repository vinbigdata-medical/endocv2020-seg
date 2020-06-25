import torch
import torch.nn as nn
import torch.nn.functional as F


class NormSoftmax(nn.Module):
    def __init__(self, in_features, out_features, temperature=0.05):
        super(NormSoftmax, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight.data)

        self.ln = nn.LayerNorm(in_features, elementwise_affine=False)
        self.temperature = temperature

    def forward(self, x):
        x = self.ln(x)
        x = torch.matmul(F.normalize(x), F.normalize(self.weight))
        x = x / self.temperature
        return x