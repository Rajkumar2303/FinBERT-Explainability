import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random

from datasets import Dataset, DatasetDict
import numpy as np
import torch

from .data import LABELS, load_source
from .leakage import connected_groups, cross_split_pairs, detect_duplicates
from .splitting import SPLITS, grouped_stratified_split, stratified_split


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def distribution(rows):
    counts = {name: sum(r["label"] == i for r in rows) for i, name in enumerate(LABELS)}
    return {"rows": len(rows), "class_counts": counts,
            "class_proportions": {k: v / len(rows) if rows else 0 for k, v in counts.items()},
            "allagree_rows": sum(r["is_allagree"] for r in rows)}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(config, archive=None):
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new --output")
    seed_everything(config.seed)
    rows, source = load_source(config, archive)
    initial = stratified_split(rows, config.fractions, config.seed)
    pairs = detect_duplicates(rows, config.near_duplicate_threshold)
    groups = connected_groups(len(rows), pairs)
    final = (grouped_stratified_split(rows, groups, config.fractions, config.seed)
             if cross_split_pairs(pairs, initial) else initial)
    remaining = cross_split_pairs(pairs, final)
    if remaining:
        raise RuntimeError("Leakage remains after grouping")
    for i, row in enumerate(rows):
        row.update(split=final[i], duplicate_group=f"group-{groups[i]:05d}")
    output.mkdir(parents=True, exist_ok=True)
    datasets = DatasetDict()
    stats = {"overall": distribution(rows), "splits": {}}
    for name, fraction in zip(SPLITS, config.fractions):
        subset = [r for r in rows if r["split"] == name]
        datasets[name] = Dataset.from_list(subset)
        with (output / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in subset:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stats["splits"][name] = {**distribution(subset), "target_fraction": fraction,
            "actual_fraction": len(subset)/len(rows), "row_deviation_from_target": len(subset)-fraction*len(rows),
            "allagree_distribution": distribution([r for r in subset if r["is_allagree"]])}
    datasets.save_to_disk(str(output / "dataset"))
    enriched = [{**p, "left_id": rows[p["left"]]["row_id"], "right_id": rows[p["right"]]["row_id"],
                 "label_conflict": rows[p["left"]]["label"] != rows[p["right"]]["label"],
                 "initial_splits": [initial[p["left"]], initial[p["right"]]],
                 "final_split": final[p["left"]]} for p in pairs]
    write_json(output / "leakage_report.json", {
        "normalization": "Unicode NFKC, casefold, collapse whitespace",
        "near_method": "character-within-word TF-IDF 3-5 grams, cosine similarity",
        "threshold": config.near_duplicate_threshold,
        "exact_pairs": sum(p["kind"] == "exact" for p in pairs),
        "near_pairs": sum(p["kind"] == "near" for p in pairs),
        "initial_cross_split_pairs": cross_split_pairs(enriched, initial),
        "final_cross_split_pairs": remaining, "all_duplicate_pairs": enriched,
        "duplicate_components": len(set(groups)),
        "moved_rows": sum(a != b for a, b in zip(initial, final)),
        "limitations": "Lexical heuristic, not a guarantee against all semantic paraphrases. Transitive groups remain together."})
    write_json(output / "initial_assignments.json", {r["row_id"]: initial[i] for i, r in enumerate(rows)})
    write_json(output / "statistics.json", stats)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "metadata.json", {**source, "dataset_repo": config.dataset_repo,
        "revision": config.revision, "seed": config.seed, "label_mapping": dict(enumerate(LABELS)),
        "allagree_matching": "normalized sentence AND label; fail on unmatched allagree keys",
        "split_method": "two-stage sklearn stratification; if leakage, deterministic greedy class-balanced connected components",
        "python": platform.python_version(), "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "datasets", "scikit-learn", "huggingface-hub")}})
    write_json(output / "manifest.json", {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(output.rglob("*")) if p.is_file()})
    return stats
