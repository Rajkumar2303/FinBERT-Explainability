"""Generate the actual-results comparison from completed, verified runs."""
import json
from pathlib import Path

from finbert_explainability.phase2.artifacts import digest, read_json, verify_manifest, verify_run


def avg_sd(value):
    return f"{value['mean']:.4f} ± {value['std']:.4f}"


def duration(seconds):
    return f"{seconds / 60:.2f} min" if seconds >= 60 else f"{seconds:.2f} s"


if __name__ == "__main__":
    root = Path("artifacts/phase2")
    output = Path("PHASE2_RESULTS.md")
    if output.exists():
        raise FileExistsError(output)
    manifest = verify_manifest("artifacts/phase1")
    test_hash = digest("artifacts/phase1/test.jsonl")
    verification = read_json(root / "execution/final_verification.json")
    assert verification["status"] == "passed" and verification["phase1_manifest_sha256"] == manifest
    rows, per_class, selected, fits, resources = [], [], [], [], []
    incomplete = []
    for kind, name, model_type in [
        ("baseline", "TF-IDF + Logistic Regression", "Linear supervised baseline"),
        ("bert", "bert-base-uncased", "Fine-tuned general BERT"),
        ("finbert", "finbert-pretrain", "Fine-tuned financial BERT"),
        ("reference", "ProsusAI/finbert", "External reference; non-independent")]:
        evaluation = root / ("reference" if kind == "reference" else f"{kind}-evaluation")
        if not (evaluation / "run_manifest.json").exists():
            incomplete.append(name)
            continue
        verify_run(evaluation)
        report = read_json(evaluation / ("reference.json" if kind == "reference" else "summary.json"))
        assert report["phase1_manifest_sha256"] == manifest and report["test_sha256"] == test_hash
        measured = read_json(root / "execution" / f"{kind}.json")
        assert measured["status"] == "completed" and measured["phase1_unchanged"]
        metric = report["metrics"]
        if kind == "reference":
            f1, accuracy = f"{metric['macro_f1']:.4f}", f"{metric['accuracy']:.4f}"
            elapsed = f"No training; {duration(measured['wall_seconds'])} inference workflow"
        else:
            f1, accuracy = avg_sd(metric["macro_f1"]), avg_sd(metric["accuracy"])
            elapsed = duration(measured["wall_seconds"])
            selection = read_json(root / kind / "selection.json")
            selected.append(f"| {name} | `{json.dumps(selection['winner']['params'])}` | {selection['winner']['validation_macro_f1']:.4f} |")
            if measured["fits"]:
                final = [r for r in measured["fits"] if Path(r["directory"]).name.startswith("seed-")]
                fits.append(f"- {name}: final three fits took {duration(sum(r['wall_seconds'] for r in final))}; "
                            f"per-seed durations: " + ", ".join(f"{r['seed']}: {duration(r['wall_seconds'])}" for r in final) + ".")
        rows.append(f"| {name} | {model_type} | {f1} | {accuracy} | {elapsed} |")
        resources.append(f"| {name} | {measured['peak_allocated_bytes']/2**30:.3f} | {measured['peak_reserved_bytes']/2**30:.3f} | {len(measured['oom_adjustments'])} |")
        for label in ["negative", "neutral", "positive"]:
            values = metric["per_class"][label]
            cells = [f"{values[k]:.4f}" if kind == "reference" else avg_sd(values[k]) for k in ["precision", "recall", "f1"]]
            per_class.append(f"| {name} | {label} | " + " | ".join(cells) + " |")
    text = "\n".join([
        "# Phase 2 actual experiment results", "",
        "All reported test evaluations use the same frozen 727-row test set. Supervised models use three independent final fits (seeds 42, 123, 2026); ± denotes sample standard deviation (ddof=1). Test results were not used for hyperparameter, checkpoint, or seed selection.", "",
        "| Model | Model type | Macro-F1 | Accuracy | Training/search workflow wall time |",
        "|---|---|---:|---:|---|", *rows, "",
        "ProsusAI/finbert is a fixed pretrained external reference with Financial PhraseBank in its original fine-tuning data. Overlap with our test rows may exist, so its evaluation is not independent or a fair held-out ranking against the other workflows. A single fixed reference has no three-seed SD. Reference time is inference/loading/export time, not training.", "",
        "## Per-class test metrics", "", "| Model | Class | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|", *per_class, "",
        "## Validation-only selections", "", "| Model | Selected hyperparameters | Validation macro-F1 |",
        "|---|---|---:|", *selected, "", *fits, "",
        "## Resource measurements", "", "| Model | Peak allocated GiB | Peak reserved GiB | OOM adjustments |",
        "|---|---:|---:|---:|", *resources, "",
        "Peaks use PyTorch CUDA max_memory_allocated/max_memory_reserved across each workflow and each individual fit. They exclude other applications, driver/context memory, and display VRAM. Workflow wall times include loading, selection, fitting, checkpoint/export writes, and integrity checks; per-fit wall times include model loading and validation/export. Trainer loop runtime (including in-loop validation/checkpointing) is also saved in every training_record.json. CPU baseline has zero CUDA tensor allocation.", "",
        "Batch configuration stayed at 8 per device × 2 accumulation = effective batch 16 unless an adjustment is explicitly listed in resource records. FP16 training, length 64, pinned revisions, and original configs/phase2.json were used. A short preflight confirmed classifier updates and finite final gradients without OOM; early FP16 loss-scaling overflow/skip events were recorded in sanity logs.", "",
        "## Integrity and completion", "",
        f"- Phase 1 manifest SHA256: `{manifest}`",
        f"- Frozen test JSONL SHA256: `{test_hash}`",
        "- All successful run manifests were verified; no Phase 1 artifact or successful prior run was overwritten.",
        "- Repeated seed-42 CUDA fits produced byte-identical checkpoint weights for both BERT and FinBERT; checksums are saved in execution/*-repeatability.json.",
        "- Existing rag-env and PyTorch 2.12.1+cu126 were used. No PyTorch reinstall and no Phase 3 work.",
        "- Incomplete experiments: " + (", ".join(incomplete) if incomplete else "none") + ".", "",
        "Checkpoint models, selected hyperparameters, logs, per-seed predictions, confusion matrices, subgroup metrics, and mean/SD summaries are in artifacts/phase2/<model> and <model>-evaluation. The combined machine-readable table is artifacts/phase2/comparison/comparison.json; resource records are under artifacts/phase2/execution.", "",
        "Limitations: three seeds give a limited variance estimate, validation is reused for search/epoch selection, text is truncated at 64 tokens, and lexical duplicate grouping cannot exclude all semantic/pretraining overlap. Do not retune on these test scores.", ""])
    output.write_text(text, encoding="utf-8")
    print(text)
