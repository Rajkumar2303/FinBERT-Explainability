import numpy as np

from ..data import LABELS


def select_errors(rows, model_probabilities, count=20, threshold=0.9):
    """Select reproducibly from misclassifications; annotation fields stay blank."""
    candidates = []
    for i, row in enumerate(rows):
        detail = {key: {"predicted_label": int(p[i].argmax()), "confidence": float(p[i].max()),
                        "probabilities": dict(zip(LABELS, p[i].tolist()))} for key, p in model_probabilities.items()}
        wrong = [key for key, value in detail.items() if value["predicted_label"] != row["label"]]
        if not wrong:
            continue
        # Any paired-seed BERT/FinBERT disagreement, not a silently introduced ensemble.
        seeds = sorted(key.split("-")[1] for key in detail if key.startswith("bert-"))
        disagree = any(detail[f"bert-{s}"]["predicted_label"] != detail[f"finbert-{s}"]["predicted_label"] for s in seeds)
        confidence = max(detail[key]["confidence"] for key in wrong)
        candidates.append({"row_id": row["row_id"], "sentence": row["sentence"], "true_label": row["label"],
            "is_allagree": row["is_allagree"], "max_error_confidence": confidence,
            "confident_error": confidence >= threshold, "bert_finbert_disagreement": disagree,
            "misclassifying_models": wrong, "model_predictions": detail})
    # Reserve half the quota for disagreements, then highest-confidence errors.
    ranked = sorted(candidates, key=lambda r: (-r["max_error_confidence"], r["row_id"]))
    chosen = [r for r in ranked if r["bert_finbert_disagreement"]][:count//2]
    chosen_ids = {r["row_id"] for r in chosen}
    chosen += [r for r in ranked if r["row_id"] not in chosen_ids][:max(0, count-len(chosen))]
    for row in chosen:
        row.update(error_category="", interpretation="", comments="")
    return chosen
