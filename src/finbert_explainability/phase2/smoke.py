"""Entirely local CPU integration exercise; outputs are not scientific results."""
from zipfile import ZipFile

import torch
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification, BertTokenizer

from ..config import Config
from ..pipeline import prepare, seed_everything, write_json
from .artifacts import digest, load_split, new_output, read_json, seal, verify_manifest
from .common import close_logger, logger_for, runtime
from .neural import tune_neural
from .settings import Settings
from .workflows import code_fingerprint, compare_evaluations, evaluate_workflow, reference_workflow, train_workflow


def make_fixture(out):
    archive = out / "synthetic.zip"
    labels = ["negative", "neutral", "positive"]
    words = ["loss decline", "stable unchanged", "profit growth"]
    lines = [f"company item{i} {words[i%3]} report number{i*13}@{labels[i%3]}" for i in range(60)]
    with ZipFile(archive, "w") as stream:
        stream.writestr("Sentences_50Agree.txt", "\n".join(lines))
        stream.writestr("Sentences_AllAgree.txt", "\n".join(lines[::2]))
    config = Config(output_dir=str(out / "fixture_phase1"), archive_sha256=digest(archive), near_duplicate_threshold=1.0)
    prepare(config, archive)
    return out / "fixture_phase1"


def tiny_model(path, reference=False, extra_word="synthetic"):
    path.mkdir()
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "company", "loss", "decline", "stable", "unchanged", "profit", "growth", "report", extra_word]
    tokenizer = BertTokenizer(vocab={word: i for i, word in enumerate(words)})
    tokenizer.save_pretrained(path)
    config = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
        num_labels=3, id2label={0: "positive", 1: "negative", 2: "neutral"},
        label2id={"positive": 0, "negative": 1, "neutral": 2})
    seed_everything(7)
    model = BertForSequenceClassification(config) if reference else BertForMaskedLM(config)
    model.save_pretrained(path)
    if extra_word == "finbert":
        # Reproduce the actual upstream legacy configuration omission.
        legacy = read_json(path / "config.json")
        legacy.pop("model_type")
        write_json(path / "config.json", legacy)


def smoke_test(output):
    # Keep the real Phase 1 root protected even when exercising synthetic fixtures.
    out = new_output(output, "artifacts/phase1")
    torch.set_num_threads(1)
    phase1 = make_fixture(out)
    cfg = Settings(phase1_dir=str(phase1), baseline_c=[1.0], baseline_ngram_ranges=[[1, 1]])
    train_workflow(cfg, "baseline", out / "baseline")
    evaluate_workflow(out / "baseline", out / "baseline_evaluation", cpu=True)
    train, validation = [load_split(phase1, split) for split in ("train", "validation")]
    # Six class-balanced examples; enough for gradients, accumulation, and all labels.
    train = [next(r for r in train if r["label"] == label) for label in range(3)] * 2
    for kind in ("bert", "finbert"):
        local = out / f"tiny-{kind}"
        tiny_model(local, extra_word=kind)
        spec = {"id": str(local)}
        run = out / kind
        run.mkdir()
        logger = logger_for(run)
        try:
            winner = tune_neural(train, validation, cfg, spec, run, logger, cpu=True, smoke=True)
        finally:
            close_logger(logger)
        if digest(run / "candidate-0/model/model.safetensors") != digest(run / "seed-42/model/model.safetensors"):
            raise AssertionError("Repeated CPU training with the same seed must reproduce weights")
        if len({digest(run / f"seed-{seed}/model/model.safetensors") for seed in cfg.seeds}) != 3:
            raise AssertionError("Different seeds must independently initialize/train tiny models")
        write_json(run / "config.json", cfg.to_dict())
        write_json(run / "selection.json", {"kind": kind, "winner": winner, "seeds": cfg.seeds,
            "phase1_manifest_sha256": verify_manifest(phase1), "test_sha256": digest(phase1 / "test.jsonl"),
            "test_used_for_selection": False, "smoke_only": True})
        seal(run)
        evaluate_workflow(run, out / f"{kind}_evaluation", cpu=True)
    local_reference = out / "tiny-reference"
    tiny_model(local_reference, reference=True)
    reference_workflow(cfg, out / "reference", cpu=True, local_spec={"id": str(local_reference)})
    compare_evaluations([out / f"{k}_evaluation" for k in ("baseline", "bert", "finbert")] + [out / "reference"],
                        out / "comparison", phase1)
    write_json(out / "smoke_receipt.json", {"status": "passed", "code_fingerprint": code_fingerprint(),
        "device": "cpu", "fp16": False, "runtime": runtime(),
        "checks": ["baseline fit/select/save/reload/evaluate", "two tiny MLM-to-classifier Trainer workflows",
                   "three independent seeds per workflow", "validation checkpoint selection",
                   "prediction and metric exports", "reordered external-reference labels",
                   "same-seed identical weights", "distinct-seed distinct weights", "legacy missing-model_type config"],
        "warning": "Synthetic data and random tiny models only. No actual model accuracy or GPU performance established."})
    return out / "smoke_receipt.json"
