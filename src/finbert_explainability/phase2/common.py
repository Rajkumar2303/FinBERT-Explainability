import importlib.metadata
import json
import logging
import platform

import numpy as np
import torch

from ..data import LABELS
from ..pipeline import write_json
from .metrics import metrics


def logger_for(out):
    logger = logging.getLogger(str(out))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(out / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def close_logger(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def runtime():
    return {"python": platform.python_version(), "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {name: importlib.metadata.version(name) for name in
            ["torch", "transformers", "accelerate", "datasets", "numpy", "scikit-learn", "huggingface-hub"]}}


def export_predictions(out, name, rows, probabilities):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(rows), 3) or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid canonical prediction probabilities")
    predictions = probabilities.argmax(axis=1)
    report = metrics([r["label"] for r in rows], predictions)
    unanimous = np.array([r["is_allagree"] for r in rows], dtype=bool)
    report["subgroups"] = {}
    for subgroup, mask in [("allagree", unanimous), ("not_allagree", ~unanimous)]:
        report["subgroups"][subgroup] = (metrics(np.array([r["label"] for r in rows])[mask], predictions[mask]) if mask.any() else None)
    with (out / f"{name}_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row, pred, probs in zip(rows, predictions, probabilities):
            stream.write(json.dumps({"row_id": row["row_id"], "split": row["split"],
                "true_label": row["label"], "predicted_label": int(pred), "is_allagree": row["is_allagree"],
                "probabilities": dict(zip(LABELS, probs.tolist()))}, sort_keys=True) + "\n")
    write_json(out / f"{name}_metrics.json", report)
    return report


def choose_candidate(records):
    # Stable tie break: first configured candidate; never inspect test results.
    if not records or not all(np.isfinite(r["validation_macro_f1"]) for r in records):
        raise ValueError("Every candidate needs a finite validation score")
    return max(records, key=lambda r: r["validation_macro_f1"])
