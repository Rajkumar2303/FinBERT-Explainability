import os
import random
import time

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from scipy.stats import spearmanr
from transformers import BertForSequenceClassification


def deterministic(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def embedding_layer(model):
    if not isinstance(model, BertForSequenceClassification):
        raise TypeError("Expected BertForSequenceClassification; refusing to guess an attribution layer")
    layer = model.bert.embeddings
    if not hasattr(layer, "word_embeddings") or not hasattr(layer, "LayerNorm"):
        raise TypeError("Unexpected BERT embeddings architecture")
    return layer


def reference_ids(ids, mask, special, replacement):
    """Keep inserted specials, padding, positions and attention fixed."""
    if ids.shape != mask.shape or ids.shape != special.shape:
        raise ValueError("Input/mask/special shape mismatch")
    content = mask.bool() & ~special.bool()
    baseline = ids.clone()
    baseline[content] = replacement
    return baseline, content


def target_forward(model, target):
    if not 0 <= target < model.config.num_labels:
        raise ValueError("Target class outside model label mapping")

    def forward(ids, attention_mask, token_type_ids):
        return model(input_ids=ids, attention_mask=attention_mask,
                     token_type_ids=token_type_ids).logits[:, target]
    return forward


def completeness(attribution, input_logit, baseline_logit, floor):
    total = float(attribution.double().sum().item())
    difference = float(input_logit - baseline_logit)
    residual = total - difference
    return {"attribution_sum": total, "logit_difference": difference,
            "signed_residual": residual, "absolute_residual": abs(residual),
            "relative_residual": abs(residual) / max(abs(difference), floor)}


def converged(metrics, cfg):
    # Both tolerances must pass. The explicit floor handles near-zero denominators.
    return (metrics["absolute_residual"] <= cfg["absolute_tolerance"] and
            metrics["relative_residual"] <= cfg["relative_tolerance"])


def aggregate_words(tokens):
    """Merge consecutive ## continuations; punctuation stays separate; no retokenization."""
    words = []
    for item in tokens:
        if not item["rank_eligible"]:
            continue
        if item["token"].startswith("##") and words and words[-1]["positions"][-1] == item["position"] - 1:
            word = words[-1]
            word["text"] += item["token"][2:]
            word["positions"].append(item["position"])
            word["signed"] += item["signed"]
            word["sum_token_absolute"] += item["absolute"]
        else:
            words.append({"text": item["token"], "positions": [item["position"]],
                          "signed": item["signed"], "sum_token_absolute": item["absolute"]})
    for word in words:
        word["absolute_of_signed_sum"] = abs(word["signed"])
    return words


def explain(model, tokenizer, row, cfg, baseline_kind="pad", target=None):
    started = time.perf_counter()
    model.eval()
    if any(m.training for m in model.modules()):
        raise RuntimeError("Model contains an active training/dropout module")
    layer = embedding_layer(model)
    device = next(model.parameters()).device
    encoded = tokenizer(row["sentence"], max_length=cfg["max_length"], padding="max_length",
                        truncation=True, return_special_tokens_mask=True, return_tensors="pt")
    special = encoded.pop("special_tokens_mask").to(device)
    batch = {k: v.to(device) for k, v in encoded.items()}
    ids, mask = batch["input_ids"], batch["attention_mask"]
    types = batch.get("token_type_ids", torch.zeros_like(ids))
    if baseline_kind not in ("pad", "mask"):
        raise ValueError("Supported references: pad, mask")
    replacement = tokenizer.pad_token_id if baseline_kind == "pad" else tokenizer.mask_token_id
    if replacement is None:
        raise ValueError(f"Tokenizer has no {baseline_kind} reference token")
    baseline, content = reference_ids(ids, mask, special, replacement)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, token_type_ids=types).logits[0]
        probabilities = logits.double().softmax(-1)
    predicted = int(logits.argmax())
    target = predicted if target is None else int(target)
    forward = target_forward(model, target)
    with torch.no_grad():
        input_logit = float(logits[target])
        baseline_logit = float(forward(baseline, mask, types)[0])
    lig = LayerIntegratedGradients(forward, layer, multiply_by_inputs=True)
    attempts = []
    for steps in cfg["n_steps"]:
        attempt_start = time.perf_counter()
        scores, delta = lig.attribute(ids, baselines=baseline, additional_forward_args=(mask, types),
            n_steps=steps, method="gausslegendre", internal_batch_size=cfg["internal_batch_size"],
            return_convergence_delta=True, attribute_to_layer_input=False)
        if scores.shape != (*ids.shape, model.config.hidden_size):
            raise ValueError(f"Unexpected attribution shape {scores.shape}")
        if not torch.isfinite(scores).all() or not torch.isfinite(delta).all():
            raise FloatingPointError("Non-finite attribution/delta")
        metrics = completeness(scores, input_logit, baseline_logit, cfg["relative_floor"])
        attempts.append({"n_steps": steps, "captum_delta": float(delta.item()), **metrics,
                         "seconds": time.perf_counter() - attempt_start,
                         "converged": converged(metrics, cfg)})
        if attempts[-1]["converged"]:
            break
    signed = scores[0].sum(-1).detach().cpu().tolist()
    dimension_absolute = scores[0].abs().sum(-1).detach().cpu().tolist()
    token_ids = ids[0].tolist()
    tokens = []
    for i, (token_id, token) in enumerate(zip(token_ids, tokenizer.convert_ids_to_tokens(token_ids))):
        eligible = bool(content[0, i]) and token_id not in tokenizer.all_special_ids
        tokens.append({"position": i, "token_id": token_id, "token": token,
            "baseline_token_id": int(baseline[0, i]), "attention_mask": int(mask[0, i]),
            "inserted_special": bool(special[0, i]), "rank_eligible": eligible,
            "signed": signed[i], "absolute": abs(signed[i]),
            "sum_absolute_embedding_dimensions": dimension_absolute[i]})
    ranking = sorted([t for t in tokens if t["rank_eligible"]], key=lambda t: (-t["absolute"], t["position"]))
    return {"row_id": row["row_id"], "sentence": row["sentence"], "true_label": row["label"],
        "predicted_label": predicted, "target_class": target, "probabilities": probabilities.cpu().tolist(),
        "logits": logits.cpu().tolist(), "target_logit": input_logit, "baseline_target_logit": baseline_logit,
        "baseline": {"kind": baseline_kind, "replacement_id": replacement,
            "attention_mask": "unchanged from original; replaced content remains attended",
            "layer": "model.bert.embeddings output (after position/type embeddings and LayerNorm)",
            "path": "straight line between complete embedding-layer outputs",
            "preserves": "CLS, SEP, original padding, sequence length, positions, token types"},
        "attribution_shape": list(scores.shape), "tokens": tokens, "words": aggregate_words(tokens),
        "ranking_positions": [t["position"] for t in ranking], "attempts": attempts,
        "converged": attempts[-1]["converged"],
        "truncated": len(tokenizer(row["sentence"], truncation=False)["input_ids"]) > cfg["max_length"],
        "runtime_seconds": time.perf_counter() - started}


def sensitivity(a, b):
    if a["row_id"] != b["row_id"] or a["target_class"] != b["target_class"]:
        raise ValueError("Sensitivity requires the same example and target")
    aa, bb = [t for t in a["tokens"] if t["rank_eligible"]], [t for t in b["tokens"] if t["rank_eligible"]]
    if [t["token_id"] for t in aa] != [t["token_id"] for t in bb]:
        raise ValueError("Sensitivity token mismatch")
    x, y = np.array([t["absolute"] for t in aa]), np.array([t["absolute"] for t in bb])
    rho = float(spearmanr(x, y).statistic) if len(x) > 1 and np.ptp(x) and np.ptp(y) else None
    k = min(5, len(aa))
    top_a, top_b = set(a["ranking_positions"][:k]), set(b["ranking_positions"][:k])
    return {"row_id": a["row_id"], "spearman_absolute_rank": rho, "top_k": k,
            "top_k_jaccard": len(top_a & top_b) / len(top_a | top_b) if top_a | top_b else None,
            "sign_change_fraction": float(np.mean([np.sign(i["signed"]) != np.sign(j["signed"]) for i,j in zip(aa,bb)])) if aa else None,
            "both_converged": a["converged"] and b["converged"]}
