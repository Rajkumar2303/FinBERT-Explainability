import numpy as np

from .io import validate_probabilities


def calibration(labels, probabilities, bins=(10, 15)):
    """Top-label, equal-width ECE; multiclass unhalved Brier in [0,2]."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Nonempty canonical integer labels required")
    p = validate_probabilities(probabilities, len(labels))
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == labels
    onehot = np.eye(3)[labels.astype(int)]
    result = {"brier": float(np.mean(np.sum((p-onehot)**2, axis=1))),
              "nll": float(-np.mean(np.log(np.clip(p[np.arange(len(labels)), labels.astype(int)], 1e-15, 1)))),
              "mean_confidence": float(confidence.mean()), "accuracy": float(correct.mean()),
              "nll_clip_epsilon": 1e-15, "binning": "[lower,upper), last bin includes 1.0", "bins": {}}
    for count in bins:
        if not isinstance(count, int) or count < 1:
            raise ValueError("Bin count must be a positive integer")
        indices = np.minimum((confidence*count).astype(int), count-1)
        table, ece = [], 0.0
        for i in range(count):
            mask = indices == i
            n = int(mask.sum())
            acc = float(correct[mask].mean()) if n else None
            conf = float(confidence[mask].mean()) if n else None
            if n:
                ece += n/len(labels) * abs(acc-conf)
            table.append({"bin": i, "lower": i/count, "upper": (i+1)/count,
                          "count": n, "accuracy": acc, "mean_confidence": conf})
        result[f"ece_{count}"] = float(ece)
        result["bins"][str(count)] = table
    return result


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("Exactly three finite seed measurements required")
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "n": 3, "ddof": 1}
