import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def fit_probe(features: np.ndarray, labels: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=42)
    classifier.fit(scaler.transform(features), labels)
    return scaler, classifier
