import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ..data import LABELS


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("Nonempty equally sized labels required")
    if not np.isin(y_true, [0, 1, 2]).all() or not np.isin(y_pred, [0, 1, 2]).all():
        raise ValueError("Labels must use canonical IDs 0, 1, 2")
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {"macro_f1": float(f.mean()), "accuracy": float(accuracy_score(y_true, y_pred)),
            "per_class": {label: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(support[i])} for i, label in enumerate(LABELS)},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
            "label_order": LABELS, "confusion_matrix_axes": "rows=true, columns=predicted"}


def trainer_metrics(prediction):
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    result = metrics(prediction.label_ids, np.argmax(logits, axis=-1))
    return {k: result[k] for k in ("macro_f1", "accuracy")}


def canonical_probabilities(logits, id2label):
    mapping = {int(k): str(v).lower() for k, v in id2label.items()}
    if set(mapping) != {0, 1, 2} or set(mapping.values()) != set(LABELS):
        raise ValueError(f"Unknown/ambiguous model label mapping: {id2label}")
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("Expected three-class logits")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    source_for_target = [next(i for i, label in mapping.items() if label == target) for target in LABELS]
    return probabilities[:, source_for_target]


def summarize(reports):
    if len(reports) != 3:
        raise ValueError("Three seed reports required for sample standard deviation")
    def summary(values):
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    return {"n_seeds": 3, "std_ddof": 1,
        **{key: summary([r[key] for r in reports]) for key in ("macro_f1", "accuracy")},
        "per_class": {label: {key: summary([r["per_class"][label][key] for r in reports])
                              for key in ("precision", "recall", "f1")} for label in LABELS},
        "confusion_matrix_mean": np.mean([r["confusion_matrix"] for r in reports], axis=0).tolist(),
        "confusion_matrix_std": np.std([r["confusion_matrix"] for r in reports], axis=0, ddof=1).tolist(),
        "label_order": LABELS, "confusion_matrix_axes": "rows=true, columns=predicted"}
