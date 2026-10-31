import gc

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig

from ..data import LABELS
from .io import validate_probabilities


def predict(spec, texts, cpu=False):
    """Load frozen checkpoint, eval/inference_mode only; canonical logits/probs."""
    if spec["kind"] == "baseline":
        model = joblib.load(spec["checkpoint"])
        if list(model.classes_) != [0, 1, 2]:
            raise ValueError("Baseline checkpoint class mapping mismatch")
        features = model["tfidf"].transform(texts)
        p = model["classifier"].predict_proba(features)
        # Visibility compares the actual sparse feature representation.
        representations = [(tuple(r.indices), tuple(np.round(r.data, 12))) for r in features]
        return validate_probabilities(p, len(texts)), None, representations, [False]*len(texts)
    cfg = BertConfig.from_pretrained(spec["checkpoint"], local_files_only=True)
    mapping = {int(i): label.lower() for i, label in cfg.id2label.items()}
    if set(mapping) != {0, 1, 2} or set(mapping.values()) != set(LABELS):
        raise ValueError("Neural checkpoint class mapping is not a permutation of canonical labels")
    columns = [next(i for i, n in mapping.items() if n == label) for label in LABELS]
    tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], config=cfg, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(spec["checkpoint"], config=cfg,
        local_files_only=True, attn_implementation="eager")
    model.requires_grad_(False)
    device = "cpu" if cpu else "cuda"
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the supported --cpu option")
    model.to(device).eval()
    full = tokenizer(texts, truncation=False, padding=False)["input_ids"]
    truncated = [len(ids) > spec["max_length"] for ids in full]
    representations = [tuple(ids) for ids in tokenizer(texts, truncation=True,
                         padding=False, max_length=spec["max_length"])["input_ids"]]
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(texts), spec["batch_size"]):
            batch = tokenizer(texts[i:i+spec["batch_size"]], padding=True, truncation=True,
                              max_length=spec["max_length"], return_tensors="pt").to(device)
            chunks.append(model(**batch).logits.float().cpu().numpy()[:, columns])
    logits = np.concatenate(chunks)
    exp = np.exp(logits.astype(float)-logits.max(axis=1, keepdims=True))
    p = exp/exp.sum(axis=1, keepdims=True)
    del model, tokenizer
    gc.collect()
    if not cpu:
        torch.cuda.empty_cache()
    return validate_probabilities(p, len(texts)), logits, representations, truncated
