import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

from model import CBOW, SkipGram
from utils import get_device, ResultLogger


def build_vocab(corpus):
    words = corpus.split()
    counter = Counter(words)
    vocab = {word: i for i, (word, _) in enumerate(counter.most_common())}
    return vocab


def make_cbow_data(corpus, vocab, window_size=2):
    words = corpus.split()
    data = []
    for i in range(window_size, len(words) - window_size):
        context = [vocab[words[i + j]] for j in range(-window_size, window_size + 1) if j != 0]
        target = vocab[words[i]]
        data.append((context, target))
    return data


def main():
    device = get_device()
    print(f"Device: {device}")

    corpus = "the quick brown fox jumps over the lazy dog the cat sits on the mat the dog barks at the cat"

    vocab = build_vocab(corpus)
    vocab_size = len(vocab)

    config = {
        "model": "CBOW",
        "vocab_size": vocab_size,
        "embedding_dim": 50,
        "window_size": 2,
        "epochs": 100,
        "learning_rate": 0.01,
        "optimizer": "Adam",
        "device": str(device),
    }

    print(f"Vocab size: {vocab_size}")

    data = make_cbow_data(corpus, vocab, window_size=2)
    model = CBOW(vocab_size, config["embedding_dim"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    logger = ResultLogger(results_dir="results", model_name="Word2Vec_CBOW")

    for epoch in range(1, config["epochs"] + 1):
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

        avg_loss = total_loss / len(data)
        logger.log(epoch=epoch, train_loss=avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

    logger.save_all(model=model, config=config)

    # 학습된 임베딩을 npy로 저장
    word_embeddings = model.embeddings.weight.data.cpu().numpy()
    import numpy as np
    np.save(os.path.join("results", "embeddings.npy"), word_embeddings)
    print(f"임베딩 저장: results/embeddings.npy (shape: {word_embeddings.shape})")


if __name__ == "__main__":
    main()
