"""Short real BERT CUDA/FP16 training using only frozen training rows."""
from dataclasses import replace
from pathlib import Path
import shutil
import math
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from finbert_explainability.pipeline import seed_everything, write_json
from finbert_explainability.phase2.artifacts import digest, load_split, new_output, read_json, verify_manifest
from finbert_explainability.phase2.neural import arguments, load_bert_config, tokenize
from finbert_explainability.phase2.settings import Settings
from finbert_explainability.phase2.workflows import training_gate


if __name__ == "__main__":
    assert Path(sys.executable).parent.name == "rag-env", sys.executable
    assert torch.cuda.is_available()
    assert "RTX 4070 Laptop GPU" in torch.cuda.get_device_name(0)
    cfg = Settings.load("configs/phase2.json")
    before = verify_manifest(cfg.phase1_dir)
    assert before == read_json("artifacts/phase2/audit/audit.json")["phase1_manifest_sha256"]
    assert digest(Path(cfg.phase1_dir) / "test.jsonl") == read_json(Path(cfg.phase1_dir) / "manifest.json")["test.jsonl"]
    training_gate(True, "artifacts/phase2/smoke-final/smoke_receipt.json")
    out = new_output("artifacts/phase2/cuda-sanity-confirmed", cfg.phase1_dir)
    seed_everything(42)
    spec = cfg.models["bert"]
    kwargs = {"revision": spec["revision"], "cache_dir": cfg.cache_dir}
    config = load_bert_config(spec, kwargs, classification=True)
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], config=config, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(spec["id"], config=config,
        attn_implementation="eager", **kwargs)
    params = {"learning_rate": cfg.learning_rates[0], "epochs": cfg.epochs[0]}
    args = arguments(out / "checkpoints", cfg, params, 42)
    args = replace(args, max_steps=12, eval_strategy="no", save_strategy="no",
                   load_best_model_at_end=False, logging_steps=1)
    data = tokenize(load_split(cfg.phase1_dir, "train")[:32], tokenizer, cfg.max_length)
    trainer = Trainer(model=model, args=args, train_dataset=data, processing_class=tokenizer)
    initial_classifier = model.classifier.weight.detach().cpu().clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    norms = [r["grad_norm"] for r in trainer.state.log_history if "grad_norm" in r]
    assert math.isfinite(norms[-1]), "FP16 scaling did not settle to finite gradients"
    assert not torch.equal(initial_classifier, model.classifier.weight.detach().cpu()), "Classifier did not update"
    report = {"status": "passed", "optimizer_steps": trainer.state.global_step,
        "wall_seconds": time.perf_counter()-start, "interpreter": sys.executable,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "fp16": args.fp16,
        "per_device_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "oom": False,
        "phase1_manifest_sha256": before, "test_sha256": digest(Path(cfg.phase1_dir) / "test.jsonl"),
        "phase1_unchanged": verify_manifest(cfg.phase1_dir) == before,
        "free_disk_bytes": shutil.disk_usage(out).free, "classifier_updated": True,
        "final_grad_norm_finite": True, "early_fp16_overflow_steps": sum(not math.isfinite(x) for x in norms),
        "log_history": trainer.state.log_history}
    write_json(out / "sanity.json", report)
    print(report)
