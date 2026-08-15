import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score

def probe_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predicted = scores >= 0.5
    return {"auroc": float(roc_auc_score(labels, scores)), "auprc": float(average_precision_score(labels, scores)), "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)), "f1": float(f1_score(labels, predicted)), "brier": float(brier_score_loss(labels, scores))}
