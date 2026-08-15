import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score

from citation_faithfulness.behavioral.experiment import model_slug
from citation_faithfulness.behavioral.statements import normalize
from citation_faithfulness.utils import ARTIFACTS, write_json


def probe_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predicted = scores >= 0.5
    return {"auroc": float(roc_auc_score(labels, scores)), "auprc": float(average_precision_score(labels, scores)), "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)), "f1": float(f1_score(labels, predicted)), "brier": float(brier_score_loss(labels, scores))}


def grouped_bootstrap(frame: pd.DataFrame, metric, resamples: int = 1000) -> tuple[float, float]:
    groups = frame.question_id.unique(); rng = np.random.default_rng(42); values = []
    for _ in range(resamples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        pieces = [frame[frame.question_id == group] for group in sampled]; sample = pd.concat(pieces, ignore_index=True)
        if sample.label.nunique() == 2: values.append(metric(sample.label, sample.score))
    quantiles = np.quantile(values, [0.025, 0.975])
    return float(quantiles[0]), float(quantiles[1])


def evaluate(model_id: str, force: bool = False) -> Path:
    directory = ARTIFACTS / "probes" / model_slug(model_id); output = directory / "test_predictions.parquet"
    if output.exists() and not force: return output
    selected = json.loads((directory / "selection.json").read_text())["selected_layer"]
    frame = pd.read_parquet(directory / "features.parquet").merge(pd.read_parquet(ARTIFACTS / "probes" / "splits.parquet"), on="question_id")
    test = frame[(frame.layer == selected) & (frame.split == "test")].copy(); scaler = joblib.load(directory / "scaler.joblib"); classifier = joblib.load(directory / "classifier.joblib")
    test["score"] = classifier.predict_proba(scaler.transform(np.stack(test.feature)))[:, 1]; test.to_parquet(output, index=False)
    metrics: dict[str, object] = dict(probe_metrics(test.label.to_numpy(), test.score.to_numpy())); metrics["auroc_ci"] = grouped_bootstrap(test, roc_auc_score); metrics["auprc_ci"] = grouped_bootstrap(test, average_precision_score)
    write_json(directory / "metrics.json", metrics); return output


def _jaccard(left: str, right: str) -> float:
    a, b = set(normalize(left).split()), set(normalize(right).split())
    return len(a & b) / len(a | b) if a | b else 0.0


def baselines(model_id: str, force: bool = False) -> Path:
    from sentence_transformers import SentenceTransformer
    directory = ARTIFACTS / "probes" / model_slug(model_id); output = directory / "baseline_predictions.parquet"
    if output.exists() and not force: return output
    selection = json.loads((directory / "selection.json").read_text()); layer = selection["selected_layer"]
    frame = pd.read_parquet(directory / "features.parquet").merge(pd.read_parquet(ARTIFACTS / "probes" / "splits.parquet"), on="question_id")
    frame = frame[(frame.layer == layer) & (frame.split == "test")].copy()
    frame["lexical_overlap"] = [_jaccard(a, b) for a, b in zip(frame.target_statement, frame.adversarial_document)]
    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    q = encoder.encode(frame.question.tolist(), normalize_embeddings=True); d = encoder.encode(frame.adversarial_document.tolist(), normalize_embeddings=True)
    frame["document_query_cosine"] = np.sum(q * d, axis=1)
    result = {}
    for name in ("lexical_overlap", "citation_logit_margin", "document_query_cosine"):
        result[name] = {"auroc": float(roc_auc_score(frame.label, frame[name])), "auprc": float(average_precision_score(frame.label, frame[name]))}
    frame.to_parquet(output, index=False); write_json(directory / "baseline_metrics.json", result); return output
