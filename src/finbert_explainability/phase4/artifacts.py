import hashlib
import json
from pathlib import Path

import numpy as np

from ..data import LABELS
from ..phase3.inputs import read_rows
from ..phase3.io import digest, load_predictions, read_json, require_listed, verify_manifest


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing a successful result. An interrupted write
    # is a visible invalid artifact and requires a new output directory.
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def save_record(path, record, context):
    payload = {"context": context, "record": record}
    exclusive_json(path, {"sha256": fingerprint(payload), "payload": payload})


def read_record(path, context):
    obj = read_json(path)
    if fingerprint(obj["payload"]) != obj["sha256"] or obj["payload"]["context"] != context:
        raise ValueError(f"Resume record integrity/context mismatch: {path}")
    return obj["payload"]["record"]


def preservation_snapshot():
    paths = []
    for root in ("artifacts/phase1", "artifacts/phase2", "artifacts/phase3"):
        paths.extend(p for p in Path(root).rglob("*") if p.is_file())
    for root in ("src", "tests", "configs"):
        paths.extend(p for p in Path(root).rglob("*") if p.is_file() and
                     p.suffix in (".py", ".json") and "phase4" not in str(p))
    paths.extend(p for p in Path('.').glob('*.md') if 'PHASE4' not in p.name)
    paths.append(Path("pyproject.toml"))
    return {str(p): digest(p) for p in sorted(set(paths))}


def phase4_path(path):
    path = Path(path).resolve()
    root = Path("artifacts/phase4").resolve()
    if root not in path.parents:
        raise ValueError("Outputs must be inside artifacts/phase4/<run>")
    return path


def validate_config(cfg):
    if cfg["model_kind"] not in ("finbert", "bert") or cfg["model_seed"] not in (42, 123, 2026):
        raise ValueError("Only existing BERT/FinBERT model seeds are supported")
    if cfg["n_steps"] != [50, 100, 200] or cfg["max_length"] != 64:
        raise ValueError("Experiment protocol requires steps 50/100/200 and max length 64")
    if cfg["examples_per_outcome"] != 5 or not 1 <= cfg["sensitivity_per_outcome"] <= 5:
        raise ValueError("Require 5 correct + 5 errors, with 1..5 sensitivity examples per outcome")
    if cfg["internal_batch_size"] < 1:
        raise ValueError("internal_batch_size must be positive")
    for key in ("absolute_tolerance", "relative_tolerance", "relative_floor", "probability_tolerance", "logit_tolerance"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid {key}")
    phase4_path(cfg["output_dir"])
    report = Path(cfg["report_path"]).resolve()
    if report != Path("PHASE4_RESULTS.md").resolve():
        phase4_path(report)


def load_verified(cfg):
    p1,p2,p3 = (Path(cfg[k]) for k in ("phase1_dir", "phase2_dir", "phase3_dir"))
    verify_manifest(p1,"manifest.json")
    for filename,key in (("manifest.json","phase1_manifest_sha256"),("test.jsonl","test_sha256")):
        if digest(p1/filename) != cfg[key]:
            raise ValueError(f"Frozen Phase 1 hash mismatch: {filename}")
    mapping = {str(i):n for i,n in enumerate(LABELS)}
    if read_json(p1/"metadata.json")["label_mapping"] != mapping:
        raise ValueError("Phase 1 mapping mismatch")
    kind,seed = cfg["model_kind"],cfg["model_seed"]
    run, evaluation = p2/kind,p2/f"{kind}-evaluation"
    verify_manifest(run,"run_manifest.json")
    verify_manifest(evaluation,"run_manifest.json")
    prefix=f"seed-{seed}"
    require_listed(run,["selection.json","config.json",f"{prefix}/model/config.json",f"{prefix}/model/model.safetensors",f"{prefix}/model/tokenizer.json"])
    require_listed(evaluation,["summary.json",f"{prefix}/test_predictions.jsonl"])
    selected,summary = read_json(run/"selection.json"),read_json(evaluation/"summary.json")
    for obj in (selected,summary):
        for key in ("phase1_manifest_sha256","test_sha256"):
            if obj[key] != cfg[key]:
                raise ValueError(f"Phase 2 input mismatch: {key}")
    if selected["test_used_for_selection"] is not False or seed not in selected["seeds"]:
        raise ValueError("Invalid checkpoint selection protocol")
    if summary["selection_run_manifest_sha256"] != digest(run/"run_manifest.json"):
        raise ValueError("Phase 2 selection manifest mismatch")
    if read_json(run/"config.json")["max_length"] != cfg["max_length"]:
        raise ValueError("Tokenization length mismatch")
    checkpoint=run/prefix/"model"
    model_cfg=read_json(checkpoint/"config.json")
    if model_cfg["id2label"] != mapping or model_cfg["label2id"] != {n:i for i,n in enumerate(LABELS)}:
        raise ValueError("Checkpoint label mapping mismatch")
    rows=read_rows(p1/"test.jsonl")
    if any(r["split"] != "test" for r in rows):
        raise ValueError("Wrong frozen split")
    previous=load_predictions(evaluation/prefix/"test_predictions.jsonl",rows)
    warnings=[]
    for name,expected in read_json(p3/"manifest.json").items():
        path=(p3/name).resolve()
        if p3.resolve() not in path.parents:
            raise ValueError("Unsafe Phase 3 manifest path")
        actual=digest(path)
        if actual != expected:
            if name == "human_annotation.csv":
                warnings.append({"artifact":name,"reason":"Known user-authorized annotation edits after sealing; not an inference input", "sealed_sha256":expected,"current_sha256":actual})
            else:
                raise ValueError(f"Phase 3 manifest mismatch: {name}")
    folder=p3/f"{kind}-{seed}"
    required=[f"{kind}-{seed}/inference_row_order.json",f"{kind}-{seed}/recomputed_inference.npz"]
    require_listed(p3,required,filename="manifest.json")
    order=read_json(folder/"inference_row_order.json")
    positions=[i for i,r in enumerate(order) if r["type"]=="original"]
    if [order[i]["row_id"] for i in positions] != [r["row_id"] for r in rows]:
        raise ValueError("Phase 3 original sample alignment mismatch")
    with np.load(folder/"recomputed_inference.npz") as data:
        probabilities=data["probabilities"][positions]
        logits=data["canonical_logits"][positions]
    if probabilities.shape != previous.shape or logits.shape != previous.shape or not np.isfinite(logits).all():
        raise ValueError("Phase 3 inference dimensions/nonfinite mismatch")
    if not np.allclose(probabilities,previous,atol=cfg["probability_tolerance"],rtol=0) or not np.array_equal(probabilities.argmax(1),previous.argmax(1)):
        raise ValueError("Phase 2/3 probability or class mismatch")
    return rows,checkpoint,previous,logits,warnings


def selection(rows, probabilities, count=5):
    correct,errors=[],[]
    for i in sorted(range(len(rows)),key=lambda i:rows[i]["row_id"]):
        (correct if int(probabilities[i].argmax())==rows[i]["label"] else errors).append(i)
    if min(len(correct),len(errors)) < count:
        raise ValueError("Insufficient correct/error samples for fixed policy")
    return correct[:count],errors[:count]
