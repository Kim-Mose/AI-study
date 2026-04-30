import torch
import torch.nn as nn


class CBOW(nn.Module):
    """
    Word2Vec - CBOW (Continuous Bag of Words)
    주변 단어로 중심 단어를 예측
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        # context: (batch, context_size)
        embeds = self.embeddings(context)        # (batch, context_size, embedding_dim)
        embeds = embeds.mean(dim=1)              # 주변 단어 임베딩 평균
        out = self.linear(embeds)
        return out


class SkipGram(nn.Module):
    """
    Word2Vec - Skip-gram
    중심 단어로 주변 단어를 예측
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center):
        # center: (batch,)
        embeds = self.embeddings(center)        # (batch, embedding_dim)
        out = self.linear(embeds)
        return out


if __name__ == "__main__":
    vocab_size = 1000
    embedding_dim = 100

    cbow = CBOW(vocab_size, embedding_dim)
    skipgram = SkipGram(vocab_size, embedding_dim)

    context = torch.randint(0, vocab_size, (4, 4))  # 배치 4, 윈도우 4
    center = torch.randint(0, vocab_size, (4,))

    print("CBOW:", cbow(context).shape)
    print("Skip-gram:", skipgram(center).shape)
