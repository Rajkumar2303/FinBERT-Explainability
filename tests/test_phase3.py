import json
from pathlib import Path

import numpy as np
import pytest

from finbert_explainability.phase3.calibration import calibration, mean_sd
from finbert_explainability.phase3.errors import select_errors
from finbert_explainability.phase3.io import digest, load_predictions, protected_output, require_listed, snapshot, verify_manifest, write_json
from finbert_explainability.phase3.perturbations import generate, perturb
from finbert_explainability.phase3.pipeline import robustness_metrics, subgroup_report, validate_config


@pytest.fixture
def config():
    return json.loads(Path("configs/phase3.json").read_text())


def row(text="Nokia is profitable.", label=2, agreement=True, index=0):
    return {"sentence": text, "label": label, "is_allagree": agreement, "row_id": f"row-{index}", "split": "test"}


def test_calibration_known_answers_and_boundaries():
    p = [[.5, .5, 0], [0, 1, 0]]
    c = calibration([0, 1], p)
    assert c["ece_10"] == pytest.approx(.25)
    assert c["ece_15"] == pytest.approx(.25)
    assert c["brier"] == pytest.approx(.25)
    assert c["nll"] == pytest.approx(-np.log(.5)/2)
    assert c["bins"]["10"][5]["count"] == 1
    assert c["bins"]["10"][9]["count"] == 1
    assert c["bins"]["10"][0]["accuracy"] is None
    json.dumps(c, allow_nan=False)
    assert calibration([1], [[1,0,0]])["nll"] == pytest.approx(-np.log(1e-15))
    assert calibration([1], [[1,0,0]])["brier"] == 2
    assert mean_sd([.1,.2,.3])["std"] == pytest.approx(.1)


@pytest.mark.parametrize("p", [[[.2,.2,.2]], [[-1,1,1]], [[float('nan'),0,1]], [[1,0]]])
def test_invalid_probabilities_fail(p):
    with pytest.raises(ValueError):
        calibration([0], p)


def test_prediction_alignment_and_mapping(tmp_path):
    path = tmp_path / "p.jsonl"
    rows = [row()]
    prediction = {"row_id": "row-0", "true_label": 2, "predicted_label": 2, "split": "test",
                  "is_allagree": True, "probabilities": {"negative": .1, "neutral": .1, "positive": .8}}
    path.write_text(json.dumps(prediction))
    assert load_predictions(path, rows).argmax(axis=1).tolist() == [2]
    for key, value in [("row_id", "wrong"), ("true_label", 1), ("is_allagree", False), ("predicted_label", 1)]:
        path.write_text(json.dumps({**prediction, key: value}))
        with pytest.raises(ValueError):
            load_predictions(path, rows)
    path.write_text(json.dumps(prediction)+"\n"+json.dumps(prediction))
    with pytest.raises(ValueError, match="duplicate"):
        load_predictions(path, rows)


def test_manifest_missing_tampered_and_unlisted(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"original")
    write_json(tmp_path / "run_manifest.json", {"model.bin": digest(tmp_path / "model.bin")})
    verify_manifest(tmp_path, "run_manifest.json")
    with pytest.raises(ValueError, match="not sealed"):
        require_listed(tmp_path, ["missing.json"])
    (tmp_path / "model.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        verify_manifest(tmp_path, "run_manifest.json")


def test_protect_all_earlier_artifacts(tmp_path):
    roots = [tmp_path / "phase1", tmp_path / "phase2"]
    for output in [roots[0], roots[1]/"new", tmp_path]:
        with pytest.raises(ValueError, match="read-only"):
            protected_output(output, roots, tmp_path / "report.md")
    with pytest.raises(ValueError):
        protected_output(tmp_path / "new", roots, roots[1] / "report.md")


def test_company_rule_deterministic_and_boundary(config):
    r = row("Nokia Siemens Networks has signed an agreement with Nokia Siemens Networks.")
    v = perturb(r, "company", config)
    assert v["perturbed_text"] == "Example Holdings has signed an agreement with Example Holdings."
    assert v["expected_behavior"] == "conditional_invariance" and v["perturbed_gold_label"] is None
    assert not perturb(row("NokiaLike is a different token."), "company", config)["eligible"]
    assert generate([r], config) == generate([r], config)


def test_money_preserves_signs_and_comparisons_not_dates(config):
    v = perturb(row("Profit rose from EUR -1.5 to EUR 3.0 in 2010, up 20%."), "numerical", config)
    assert v["perturbed_text"] == "Profit rose from EUR -3.0 to EUR 6.0 in 2010, up 20%."
    v = perturb(row("Profit was EUR0 .12."), "numerical", config)
    assert "EUR0.24" in v["perturbed_text"]
    assert "source_spaced_decimal_normalized" in v["review_flags"]
    assert not perturb(row("Amount EUR 1,2."), "numerical", config)["eligible"]
    assert not perturb(row("Amount EUR 1-2 million."), "numerical", config)["eligible"]
    assert not perturb(row("Revenue rose 20% in 2010."), "numerical", config)["eligible"]


def test_negation_never_assigns_reversed_gold(config):
    v = perturb(row("The company has increased its losses.", label=0), "negation", config)
    assert v["perturbed_text"] == "The company has not increased its losses."
    assert v["expected_behavior"] == "possible_sentiment_change_direction_unspecified"
    assert v["perturbed_gold_label"] is None and not v["human_validated"]
    assert not perturb(row("The company has not increased losses."), "negation", config)["eligible"]
    assert not perturb(row("The company has EUR 2 million."), "negation", config)["eligible"]


def test_small_subgroups_and_invisible_edits(config):
    groups = subgroup_report([row(), row(label=0, agreement=False, index=1)], np.array([[0,0,1],[1,0,0]]), config)
    assert "small_subgroup" in groups["allagree"]["warnings"]
    assert "small_class:negative:0" in groups["allagree"]["warnings"]
    records = [{"type": "company", "model_visible_change": False, "class_flip": False,
                "confidence_change": 0, "original_class_probability_change": 0}]
    assert robustness_metrics(records)["company"]["visible_flip_rate"] is None
    assert robustness_metrics(records)["company"]["flip_rate"] == 0


def test_error_selection_has_blank_annotations_and_disagreements():
    rows = [row(label=0, index=i) for i in range(30)]
    probabilities = {}
    for seed in [42,123,2026]:
        probabilities[f"bert-{seed}"] = np.tile([.01,.98,.01], (30,1))
        probabilities[f"finbert-{seed}"] = np.tile([.01,.01,.98], (30,1))
    selected = select_errors(rows, probabilities)
    assert len(selected) == len({r["row_id"] for r in selected}) == 20
    assert all(r["bert_finbert_disagreement"] and r["confident_error"] for r in selected)
    assert all(r[k] == "" for r in selected for k in ["error_category", "interpretation", "comments"])


def test_readonly_cpu_inference_with_tiny_local_checkpoint(tmp_path):
    from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
    from finbert_explainability.phase3.inference import predict
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "profit", "loss"]
    tokenizer = BertTokenizer(vocab={w:i for i,w in enumerate(words)})
    tokenizer.save_pretrained(tmp_path)
    cfg = BertConfig(vocab_size=len(words), hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, num_labels=3, id2label={0:"positive",1:"negative",2:"neutral"})
    BertForSequenceClassification(cfg).save_pretrained(tmp_path)
    before = snapshot([tmp_path])
    p, logits, rep, truncated = predict({"kind":"bert", "checkpoint":str(tmp_path), "max_length":8, "batch_size":2},
        ["profit "*20, "profit "*20+"loss"], cpu=True)
    assert p.shape == logits.shape == (2,3)
    assert np.allclose(p.sum(axis=1), 1) and rep[0] == rep[1] and all(truncated)
    assert snapshot([tmp_path]) == before


def test_config_validation(config):
    validate_config(config)
    with pytest.raises(ValueError):
        validate_config({**config, "monetary_scale": 1})
    with pytest.raises(ValueError):
        validate_config({**config, "seeds": [42,42,42]})
