import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTEmbedding(nn.Module):
    """
    BERT의 임베딩 = Token + Segment + Position
    """
    def __init__(self, vocab_size, d_model, max_len=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.segment = nn.Embedding(2, d_model)  # 두 문장 구분 (0, 1)
        self.position = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, segment):
        # x, segment: (batch, seq_len)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)

        embed = self.token(x) + self.segment(segment) + self.position(positions)
        return self.dropout(self.norm(embed))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class BERT(nn.Module):
    """
    BERT (2018)
    Transformer 인코더만 사용한 양방향 언어 모델
    """
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, d_ff=3072, max_len=512):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x, segment, mask=None):
        x = self.embedding(x, segment)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BERTForMLM(nn.Module):
    """Masked Language Model (MLM) 헤드"""
    def __init__(self, bert, vocab_size, d_model=768):
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, segment, mask=None):
        out = self.bert(x, segment, mask)
        return self.linear(out)


class BERTForNSP(nn.Module):
    """Next Sentence Prediction (NSP) 헤드"""
    def __init__(self, bert, d_model=768):
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(d_model, 2)

    def forward(self, x, segment, mask=None):
        out = self.bert(x, segment, mask)
        cls = out[:, 0, :]  # [CLS] 토큰의 출력 사용
        return self.linear(cls)


if __name__ == "__main__":
    vocab_size = 30000
    bert = BERT(vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=512)

    x = torch.randint(0, vocab_size, (2, 10))
    segment = torch.zeros(2, 10, dtype=torch.long)
    out = bert(x, segment)
    print(out.shape)  # (2, 10, 128)
