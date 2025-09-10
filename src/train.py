from __future__ import annotations
import argparse, os, yaml, numpy as np
from typing import Dict, List
from .utils import set_seed, save_json, Timer
from .data import load_imdb, few_shot_indices
from .eval import compute_metrics, hardest_examples
from .plotting import plot_learning_curve

# Baseline imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# DistilBERT imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch


def make_training_args(cfg):
    base = {
        "output_dir": "results/hf_runs",
        "per_device_train_batch_size": cfg["batch_size"],
        "per_device_eval_batch_size": max(16, cfg["batch_size"]),
        "num_train_epochs": cfg["epochs"],
        "learning_rate": cfg["learning_rate"],
        "weight_decay": cfg["weight_decay"],
        "warmup_ratio": cfg["warmup_ratio"],
        "evaluation_strategy": "epoch",
        "save_strategy": "no",
        "logging_strategy": "epoch",
        "report_to": [],
        "load_best_model_at_end": False,
    }
    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())  # accepted parameter names
    args = {}
    for k, v in base.items():
        key = k
        # alias for some versions
        if k == "evaluation_strategy" and k not in allowed and "eval_strategy" in allowed:
            key = "eval_strategy"
        if key in allowed:
            args[key] = v
    return TrainingArguments(**args)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    p.add_argument("--fast", action="store_true", help="Tiny quick run: sizes=[50], epochs=1, max_test_examples=1000")
    return p

def load_config(path: str, fast: bool) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if fast:
        cfg["train_sizes"] = [50]
        cfg["epochs"] = 1
        cfg["max_test_examples"] = 1000
    return cfg

def train_eval_svm(train_texts, train_labels, test_texts, test_labels, cfg) -> Dict[str, float]:
    # Vectorise
    vec = TfidfVectorizer(
        max_features=cfg["tfidf"]["max_features"],
        ngram_range=tuple(cfg["tfidf"]["ngram_range"]),
        min_df=cfg["tfidf"]["min_df"],
        stop_words="english",
        lowercase=True,
    )
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    # Train SVM
    clf = LinearSVC(C=cfg["svm"]["C"])
    clf.fit(Xtr, train_labels)
    # Predict
    preds = clf.predict(Xte)
    m = compute_metrics(test_labels, preds)
    return m, preds.tolist()

def tokenize_batch(tokenizer, texts, max_length):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length)

def train_eval_distilbert(train_texts, train_labels, test_texts, test_labels, cfg) -> Dict[str, float]:
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Prepare datasets for Trainer
    class SimpleDS(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length)
            self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_ds = SimpleDS(train_texts, train_labels, tokenizer, cfg["max_length"])
    test_ds  = SimpleDS(test_texts,  test_labels,  tokenizer, cfg["max_length"])

    args = make_training_args(cfg)

    def compute_metrics_fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_metrics(labels, preds)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics_fn,
    )
    trainer.train()
    # Evaluate
    preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    m = compute_metrics(test_labels, preds)
    return m, preds.tolist()

def main():
    args = build_argparser().parse_args()
    cfg = load_config(args.config, args.fast)
    set_seed(cfg.get("seed", 42))

    # Load data once
    (Xtr_all, ytr_all), (Xte, yte) = load_imdb(max_test_examples=cfg.get("max_test_examples", None),
                                               seed=cfg.get("seed", 42))

    # Accumulate metrics
    all_metrics = {}
    last_preds = {"svm": None, "distilbert": None}

    for n in cfg["train_sizes"]:
        # Few-shot sample per class from the full training set
        idx = few_shot_indices(ytr_all, n_per_class=int(n), seed=cfg.get("seed", 42))
        Xtr = [Xtr_all[i] for i in idx]
        ytr = [ytr_all[i] for i in idx]

        # --- Baseline SVM
        with Timer() as t:
            m_svm, preds_svm = train_eval_svm(Xtr, ytr, Xte, yte, cfg)
        m_svm["train_seconds"] = t.elapsed

        # --- DistilBERT fine-tune
        with Timer() as t:
            m_bert, preds_bert = train_eval_distilbert(Xtr, ytr, Xte, yte, cfg)
        m_bert["train_seconds"] = t.elapsed

        all_metrics[str(n)] = {"svm": m_svm, "distilbert": m_bert}
        last_preds = {"svm": preds_svm, "distilbert": preds_bert}

    os.makedirs("results", exist_ok=True)
    save_json(all_metrics, "results/metrics.json")

    # Learning-curve plots
    plot_learning_curve(all_metrics, "results/plots/learning_curve.png")

    # Hard examples from the last run (largest n)
    examples = hardest_examples(Xte, yte, last_preds["svm"], last_preds["distilbert"], k=6)
    with open("results/examples.md", "w", encoding="utf-8") as f:
        print("# Hard cases (qualitative)", file=f)
        print("", file=f)
        for i, ex in enumerate(examples, 1):
            print(f"## Example {i}", file=f)
            lab = "pos" if ex["label"]==1 else "neg"
            sp = "pos" if ex["svm_pred"]==1 else "neg"
            bp = "pos" if ex["distilbert_pred"]==1 else "neg"
            print(f"**True:** {lab} | **SVM:** {sp} | **DistilBERT:** {bp}", file=f)
            print("", file=f)
            print(ex["text"], file=f)
            print("", file=f)

    print("Done. See results/metrics.json and results/plots/learning_curve.png")

if __name__ == "__main__":
    main()
