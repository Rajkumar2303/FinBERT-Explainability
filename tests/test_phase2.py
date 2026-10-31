from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from finbert_explainability.phase2.artifacts import digest, load_split, new_output, read_json, seal, verify_manifest, verify_run
from finbert_explainability.phase2.baseline import fit_baseline
from finbert_explainability.phase2.common import choose_candidate, export_predictions
from finbert_explainability.phase2.metrics import canonical_probabilities, metrics, summarize
from finbert_explainability.phase2.neural import arguments
from finbert_explainability.phase2.settings import Settings
from finbert_explainability.phase2.workflows import train_workflow, training_gate


def rows(split="train"):
    return [{"sentence": text, "label": label, "row_id": f"{split}-{i}", "split": split, "is_allagree": i % 2 == 0}
        for i, (text, label) in enumerate([("loss decline", 0), ("stable unchanged", 1), ("profit growth", 2),
                                          ("decline loss", 0), ("unchanged stable", 1), ("growth profit", 2)])]


def test_reference_label_permutation_and_unknown_labels():
    probabilities = canonical_probabilities([[8, 1, 0], [0, 9, 1], [0, 1, 7]],
        {"0": "positive", "1": "negative", "2": "neutral"})
    assert probabilities.argmax(axis=1).tolist() == [2, 0, 1]
    assert np.allclose(probabilities.sum(axis=1), 1)
    with pytest.raises(ValueError, match="mapping"):
        canonical_probabilities([[1, 2, 3]], {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_metrics_axes_missing_class_and_sample_std():
    report = metrics([0, 0, 1], [0, 1, 1])
    assert report["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
    assert report["per_class"]["positive"]["f1"] == 0
    assert report["accuracy"] == pytest.approx(2/3)
    assert report["macro_f1"] == pytest.approx(4/9)
    reports = [dict(report, accuracy=x) for x in [0.5, 0.6, 0.7]]
    assert summarize(reports)["accuracy"]["std"] == pytest.approx(0.1)


def test_training_only_vocabulary_and_checkpoint_reload(tmp_path):
    import joblib
    model = fit_baseline(rows(), {"C": 1, "ngram_range": [1, 1]}, Settings(), 42)
    heldout = [dict(r, sentence="heldoutonlytoken " + r["sentence"]) for r in rows("validation")]
    before = dict(model["tfidf"].vocabulary_)
    p = model.predict_proba([r["sentence"] for r in heldout])
    assert "heldoutonlytoken" not in before
    assert model["tfidf"].vocabulary_ == before
    joblib.dump(model, tmp_path / "model.joblib")
    assert np.allclose(joblib.load(tmp_path / "model.joblib").predict_proba([r["sentence"] for r in heldout]), p)
    export_predictions(tmp_path, "validation", heldout, p)
    assert len((tmp_path / "validation_predictions.jsonl").read_text().splitlines()) == 6


def test_selection_tie_is_stable_and_ignores_test_scores():
    records = [{"validation_macro_f1": 0.5, "test_macro_f1": 1.0},
               {"validation_macro_f1": 0.6, "test_macro_f1": 0.0},
               {"validation_macro_f1": 0.6, "test_macro_f1": 1.0}]
    assert choose_candidate(records) is records[1]


def test_training_never_loads_test(monkeypatch, tmp_path):
    import finbert_explainability.phase2.workflows as w
    seen = []
    monkeypatch.setattr(w, "verify_manifest", lambda _: "fixed")
    monkeypatch.setattr(w, "digest", lambda _: "fixed-test-hash")
    def loader(root, name):
        assert name != "test"
        seen.append(name)
        return rows(name)
    monkeypatch.setattr(w, "load_split", loader)
    cfg = Settings(phase1_dir=str(tmp_path / "frozen"), baseline_c=[1], baseline_ngram_ranges=[[1, 1]])
    train_workflow(cfg, "baseline", tmp_path / "run")
    assert seen == ["train", "validation"]
    assert read_json(tmp_path / "run/selection.json")["test_used_for_selection"] is False
    verify_run(tmp_path / "run")


def test_full_training_gate_and_reference_prohibition(tmp_path):
    with pytest.raises(PermissionError, match="approval"):
        training_gate(False, tmp_path / "absent.json")
    (tmp_path / "receipt.json").write_text('{"status":"passed","code_fingerprint":"stale"}')
    with pytest.raises(ValueError, match="current"):
        training_gate(True, tmp_path / "receipt.json")
    with pytest.raises(ValueError, match="never"):
        train_workflow(Settings(), "reference", tmp_path / "run")


def test_output_protection_and_sealed_run(tmp_path):
    frozen = tmp_path / "phase1"
    for path in [frozen, frozen / "new", tmp_path]:
        with pytest.raises(ValueError, match="read-only"):
            new_output(path, frozen)
    run = new_output(tmp_path / "run", frozen)
    (run / "a.txt").write_text("original")
    seal(run)
    verify_run(run)
    (run / "a.txt").write_text("changed")
    with pytest.raises(ValueError, match="integrity"):
        verify_run(run)


def test_finetuning_arguments(tmp_path):
    args = arguments(tmp_path, Settings(), {"learning_rate": 2e-5, "epochs": 3}, 42, cpu=True)
    assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 16
    assert args.learning_rate == 2e-5 and args.num_train_epochs == 3
    assert not args.fp16 and args.use_cpu
    assert args.load_best_model_at_end and args.metric_for_best_model == "macro_f1"
    assert args.full_determinism and args.dataloader_num_workers == 0


def test_legacy_finbert_config_without_model_type(tmp_path):
    from finbert_explainability.phase2.neural import load_bert_config
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 16, "num_hidden_layers": 1,
        "num_attention_heads": 2, "intermediate_size": 32, "vocab_size": 20}))
    cfg = load_bert_config({"id": str(tmp_path)}, {"local_files_only": True}, classification=True)
    assert cfg.model_type == "bert" and cfg.num_labels == 3
    assert cfg.id2label == {0: "negative", 1: "neutral", 2: "positive"}


def test_manifest_rejects_changed_split(tmp_path):
    from finbert_explainability.pipeline import write_json
    from finbert_explainability.data import LABELS
    names = ["train.jsonl", "validation.jsonl", "test.jsonl", "config.json", "metadata.json",
             "leakage_report.json", "initial_assignments.json", "statistics.json"]
    for name in names:
        (tmp_path / name).write_text("{}")
    write_json(tmp_path / "metadata.json", {"label_mapping": dict(enumerate(LABELS))})
    write_json(tmp_path / "manifest.json", {name: digest(tmp_path / name) for name in names})
    verify_manifest(tmp_path)
    (tmp_path / "test.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="test.jsonl"):
        verify_manifest(tmp_path)


def test_comparison_rejects_different_test_sets(tmp_path):
    from finbert_explainability.pipeline import write_json
    from finbert_explainability.phase2.workflows import compare_evaluations
    runs = []
    for kind in ["baseline", "bert", "finbert"]:
        path = tmp_path / kind
        path.mkdir()
        write_json(path / "summary.json", {"kind": kind, "phase1_manifest_sha256": "fixed",
                                          "test_sha256": "other" if kind == "bert" else "same"})
        seal(path)
        runs.append(path)
    with pytest.raises(ValueError, match="same frozen"):
        compare_evaluations(runs, tmp_path / "comparison", tmp_path / "frozen")


@pytest.mark.parametrize("kwargs", [{"seeds": [42, 42, 42]}, {"max_length": 128},
    {"learning_rates": [0]}, {"fp16": False}, {"models": {"bert": {"revision": "main"}}}])
def test_invalid_phase2_config(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
