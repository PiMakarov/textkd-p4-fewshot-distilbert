from __future__ import annotations
from typing import List, Dict, Tuple
from datasets import load_dataset
import numpy as np

def load_imdb(max_test_examples: int | None = 2000, seed: int = 42):
    ds = load_dataset("imdb")
    train = ds["train"]
    test = ds["test"]
    # Optional: cap test set for faster eval on CPU
    if max_test_examples is not None:
        test = test.shuffle(seed=seed).select(range(min(max_test_examples, len(test))))
    X_train = list(train["text"])
    y_train = list(train["label"])  # 0=neg, 1=pos
    X_test = list(test["text"])
    y_test = list(test["label"])
    return (X_train, y_train), (X_test, y_test)

def few_shot_indices(labels: List[int], n_per_class: int, seed: int = 42) -> List[int]:
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    idxs = []
    for c in sorted(set(labels)):
        cand = np.where(labels == c)[0]
        chosen = rng.choice(cand, size=min(n_per_class, len(cand)), replace=False)
        idxs.extend(chosen.tolist())
    rng.shuffle(idxs)
    return idxs
