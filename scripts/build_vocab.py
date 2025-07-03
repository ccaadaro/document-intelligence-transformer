import os
from collections import Counter
import pickle

def build_vocab_from_files(data_dir, min_freq=2, max_size=10000):
    counter = Counter()
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".txt"):
                with open(os.path.join(root, fname), encoding='utf-8') as f:
                    for line in f:
                        tokens = line.strip().lower().split()
                        counter.update(tokens)

    # Reservamos tokens especiales
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_size:
            break
        vocab[word] = idx
        idx += 1

    return vocab

if __name__ == "__main__":
    vocab = build_vocab_from_files("data/train_texts/")
    with open("training/tokenizer.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print(f"✅ Vocabulario guardado: {len(vocab)} tokens.")