from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from citation_faithfulness.behavioral.experiment import model_slug
from citation_faithfulness.utils import ARTIFACTS, write_json, write_manifest


def fit_probe(features: np.ndarray, labels: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=42)
    classifier.fit(scaler.transform(features), labels)
    return scaler, classifier


def train_layers(model_id: str, force: bool = False) -> Path:
    directory = ARTIFACTS / "probes" / model_slug(model_id); output = directory / "layer_validation.parquet"
    if output.exists() and not force: return output
    features = pd.read_parquet(directory / "features.parquet").merge(pd.read_parquet(ARTIFACTS / "probes" / "splits.parquet"), on="question_id")
    records = []; fitted = {}
    for layer, rows in features.groupby("layer"):
        train = rows[rows.split == "train"]; validation = rows[rows.split == "validation"]
        if train.label.nunique() < 2 or validation.label.nunique() < 2: continue
        scaler, classifier = fit_probe(np.stack(train.feature), train.label.to_numpy())
        scores = classifier.predict_proba(scaler.transform(np.stack(validation.feature)))[:, 1]
        records.append({"layer": int(layer), "relative_depth": int(layer) / max(1, features.layer.max()), "validation_auroc": float(roc_auc_score(validation.label, scores))})
        fitted[int(layer)] = (scaler, classifier)
    result = pd.DataFrame(records).sort_values(["validation_auroc", "layer"], ascending=[False, True])
    if result.empty: raise RuntimeError("No layer had both probe classes in train and validation")
    selected = int(result.iloc[0].layer); directory.mkdir(parents=True, exist_ok=True); result.sort_values("layer").to_parquet(output, index=False)
    joblib.dump(fitted[selected][0], directory / "scaler.joblib"); joblib.dump(fitted[selected][1], directory / "classifier.joblib")
    write_json(directory / "selection.json", {"selected_layer": selected, "relative_depth": selected / max(1, features.layer.max())}); write_manifest(f"probe train --model {model_id}"); return output
