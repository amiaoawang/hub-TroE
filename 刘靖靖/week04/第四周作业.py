import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MutiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = MutiHeadAttention(d_model, num_heads, dropout)

        act_layer = nn.GELU if activation == "gelu" else nn.ReLU
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        normed_x = self.norm1(x)
        attn_out = self.self_attn(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout(attn_out)

        normed_x = self.norm2(x)
        ff_out = self.ff(normed_x)
        x = x + ff_out

        return x
    
def main():
    batch_size , seq_len, d_model = 8, 12, 512
    num_heads, d_ff, dropout = 8, 2048, 0.1

    block = TransformerBlock(d_model, num_heads, d_ff, dropout)

    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    mask = mask.unsqueeze(0).unsqueeze(0)

    out = block(x, mask)

    print(out.shape)

if __name__ == "__main__":
    main()
