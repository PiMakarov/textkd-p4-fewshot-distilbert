from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return {"accuracy": acc, "macro_f1": f1}

def hardest_examples(texts: List[str], y_true: List[int], y_pred_a: List[int], y_pred_b: List[int], k: int = 5) -> List[Dict]:
    items = []
    for i, (t, yt, ya, yb) in enumerate(zip(texts, y_true, y_pred_a, y_pred_b)):
        wrong_both = (ya != yt) and (yb != yt)
        disagree = ya != yb
        if wrong_both or disagree:
            items.append({"text": t[:500].replace("\n", " "), "label": int(yt), "svm_pred": int(ya), "distilbert_pred": int(yb)})
    return items[:k]
