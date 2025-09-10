from __future__ import annotations
from typing import Dict, List
import os
import matplotlib.pyplot as plt

def plot_learning_curve(metrics: dict, out_path: str) -> None:
    # metrics structure: {train_size: {"svm": {...}, "distilbert": {...}}}
    sizes = sorted(int(s) for s in metrics.keys())
    acc_svm = [metrics[str(s)]["svm"]["accuracy"] for s in sizes]
    f1_svm  = [metrics[str(s)]["svm"]["macro_f1"] for s in sizes]
    acc_bert= [metrics[str(s)]["distilbert"]["accuracy"] for s in sizes]
    f1_bert = [metrics[str(s)]["distilbert"]["macro_f1"] for s in sizes]

    # Accuracy plot
    plt.figure()
    plt.plot(sizes, acc_svm, marker="o", label="TF-IDF + SVM (Accuracy)")
    plt.plot(sizes, acc_bert, marker="o", label="DistilBERT (Accuracy)")
    plt.xlabel("Train size per class")
    plt.ylabel("Accuracy")
    plt.title("Learning curve — Accuracy")
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path.replace(".png", "_acc.png"), bbox_inches="tight")
    plt.close()

    # Macro-F1 plot
    plt.figure()
    plt.plot(sizes, f1_svm, marker="o", label="TF-IDF + SVM (Macro-F1)")
    plt.plot(sizes, f1_bert, marker="o", label="DistilBERT (Macro-F1)")
    plt.xlabel("Train size per class")
    plt.ylabel("Macro-F1")
    plt.title("Learning curve — Macro-F1")
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path.replace(".png", "_f1.png"), bbox_inches="tight")
    plt.close()

    # Combined: keep README simple by copying F1 figure as the main plot name.
    from shutil import copyfile
    copyfile(out_path.replace(".png", "_f1.png"), out_path)
