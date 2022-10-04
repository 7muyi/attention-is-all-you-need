import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    '''Scaled Dot-Product Attention'''

    # temperature:sqrt(dk)
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention,self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # qxk.T/sqrt(dk)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # 掩码，将扩展的内容变为无穷小，在softmax后使其结果中接近0
        if mask is not None:
            attn = attn.masked_fill(mask==0, -1e9)

        # 注意力矩阵
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn       


