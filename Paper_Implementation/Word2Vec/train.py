import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

from model import CBOW, SkipGram


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def build_vocab(corpus):
    """단어 사전 만들기"""
    words = corpus.split()
    counter = Counter(words)
    vocab = {word: i for i, (word, _) in enumerate(counter.most_common())}
    return vocab


def make_cbow_data(corpus, vocab, window_size=2):
    """CBOW 학습 데이터 생성: (주변 단어들, 중심 단어)"""
    words = corpus.split()
    data = []
    for i in range(window_size, len(words) - window_size):
        context = [vocab[words[i + j]] for j in range(-window_size, window_size + 1) if j != 0]
        target = vocab[words[i]]
        data.append((context, target))
    return data


def make_skipgram_data(corpus, vocab, window_size=2):
    """Skip-gram 학습 데이터 생성: (중심 단어, 주변 단어)"""
    words = corpus.split()
    data = []
    for i in range(window_size, len(words) - window_size):
        center = vocab[words[i]]
        for j in range(-window_size, window_size + 1):
            if j != 0:
                context = vocab[words[i + j]]
                data.append((center, context))
    return data


def train_cbow(model, data, vocab_size, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for context, target in data:
            context = torch.tensor([context], dtype=torch.long).to(device)
            target = torch.tensor([target], dtype=torch.long).to(device)

            pred = model(context)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss / len(data):.4f}")


def main():
    device = get_device()
    print(f"Device: {device}")

    # 예시 코퍼스 (실제로는 큰 데이터셋 사용)
    corpus = "the quick brown fox jumps over the lazy dog the cat sits on the mat the dog barks at the cat"

    vocab = build_vocab(corpus)
    vocab_size = len(vocab)
    embedding_dim = 50

    print(f"Vocab size: {vocab_size}")

    # CBOW 학습
    data = make_cbow_data(corpus, vocab, window_size=2)
    model = CBOW(vocab_size, embedding_dim).to(device)
    train_cbow(model, data, vocab_size, epochs=100, lr=0.01, device=device)

    # 학습된 임베딩 추출
    word_embeddings = model.embeddings.weight.data
    print(f"임베딩 shape: {word_embeddings.shape}")


if __name__ == "__main__":
    main()
