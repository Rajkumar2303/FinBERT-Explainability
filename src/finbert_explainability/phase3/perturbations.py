"""Conservative deterministic edits. Perturbed sentiment labels are UNKNOWN."""
from decimal import Decimal
import re

TYPES = ("company", "numerical", "negation")
CURRENCY = re.compile(r"(?P<currency>\b(?:EUR|USD|GBP|SEK|FIM)\s*)(?P<amount>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\s*\.\s*\d+)?)", re.I)
NEGATED = re.compile(r"\b(?:not|never|no|without|neither|nor)\b|n['’]t", re.I)
AUXILIARY = re.compile(r"\b(?:is|are|was|were)\b(?=\s+[A-Za-z])|\b(?:will|would|can|could|should)\b(?=\s+[A-Za-z])|\b(?:has|have|had)\b(?=\s+(?:been|increased|decreased|reported|signed|risen|fallen|grown|sold|acquired|announced|improved|declined|completed)\b)", re.I)


def perturb(row, kind, cfg):
    original = row["sentence"]
    result = {"row_id": row["row_id"], "type": kind, "original_text": original,
        "perturbed_text": None, "eligible": False, "reason": None, "edits": [],
        "expected_behavior": "possible_sentiment_change_direction_unspecified" if kind == "negation" else "conditional_invariance",
        "perturbed_gold_label": None, "human_validated": False,
        "review_flags": ["semantic_consistency_not_human_validated", "source_and_output_grammar_not_validated"]}
    if "\ufffd" in original or "Ã" in original:
        result["review_flags"].append("source_encoding_artifact")
    if kind == "company":
        matches = [(m.start(), -len(name), name, m) for name in cfg["company_names"]
                   for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", original)]
        if not matches:
            result["reason"] = "no_configured_company_name"
            return result
        _, _, name, _ = min(matches, key=lambda m: m[:3])
        regex = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)")
        text = regex.sub(cfg["replacement_company"], original)
        result["edits"] = [{"from": name, "to": cfg["replacement_company"], "occurrences": len(regex.findall(original))}]
        result["review_flags"].append("identity_brand_or_world_knowledge_may_matter")
    elif kind == "numerical":
        matches = list(CURRENCY.finditer(original))
        if not matches:
            result["reason"] = "no_unambiguous_currency_prefixed_amount"
            return result
        # Avoid partial parsing of ambiguous decimal-comma amounts, percentages or ranges.
        if any(re.match(r",\d|\s*(?:%|[-–]\s*\d)", original[m.end():]) for m in matches):
            result["reason"] = "ambiguous_amount_format_or_range"
            return result
        scale = Decimal(str(cfg["monetary_scale"]))
        def replace(match):
            number = match["amount"]
            normalized = re.sub(r"\s|,", "", number)
            replacement = format(Decimal(normalized)*scale, "f")
            if re.search(r"\s", number):
                result["review_flags"].append("source_spaced_decimal_normalized")
            result["edits"].append({"from": match[0], "to": match["currency"]+replacement})
            return match["currency"] + replacement
        text = CURRENCY.sub(replace, original)
        result["review_flags"].append("only_currency_prefixed_amounts_scaled_other_quantities_unchanged")
    elif kind == "negation":
        if NEGATED.search(original):
            result["reason"] = "existing_negation_avoids_double_negation"
            return result
        match = AUXILIARY.search(original)
        if not match:
            result["reason"] = "no_supported_auxiliary_insertion"
            return result
        # "can not" is grammatical; do not concatenate to "cannot" mid-token.
        text = original[:match.end()] + " not" + original[match.end():]
        result["edits"] = [{"from": match[0], "to": match[0]+" not", "offset": match.start()}]
        result["review_flags"].append("negation_scope_grammar_and_financial_sentiment_need_review")
    else:
        raise ValueError(f"Unknown perturbation type: {kind}")
    if text == original:
        result["reason"] = "no_effective_text_change"
        return result
    result.update(eligible=True, perturbed_text=text, reason="rule_applied")
    result["review_flags"] = sorted(set(result["review_flags"]))
    return result


def generate(rows, cfg):
    return [perturb(row, kind, cfg) for kind in TYPES for row in rows]
