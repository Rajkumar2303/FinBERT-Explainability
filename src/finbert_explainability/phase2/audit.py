from collections import Counter
from pathlib import Path

from ..config import Config
from ..leakage import connected_groups, cross_split_pairs, detect_duplicates
from ..pipeline import distribution, write_json
from ..splitting import SPLITS, grouped_stratified_split, stratified_split
from .artifacts import load_split, new_output, read_json, verify_manifest


def audit_phase1(root, output):
    before = verify_manifest(root)
    rows = sorted([r for s in SPLITS for r in load_split(root, s)], key=lambda r: r["source_index"])
    if [r["source_index"] for r in rows] != list(range(len(rows))) or len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("Missing or repeated Phase 1 row IDs/source indices")
    cfg = Config.load(Path(root) / "config.json")
    pairs = detect_duplicates(rows, cfg.near_duplicate_threshold)
    groups = connected_groups(len(rows), pairs)
    if any(r["duplicate_group"] != f"group-{g:05d}" for r, g in zip(rows, groups)):
        raise ValueError("Recomputed duplicate components differ")
    initial = stratified_split(rows, cfg.fractions, cfg.seed)
    original = read_json(Path(root) / "initial_assignments.json")
    if any(original[r["row_id"]] != a for r, a in zip(rows, initial)):
        raise ValueError("Initial stratification did not reproduce")
    first = grouped_stratified_split(rows, groups, cfg.fractions, cfg.seed)
    second = grouped_stratified_split(rows, groups, cfg.fractions, cfg.seed)
    expected = first if cross_split_pairs(pairs, initial) else initial
    saved = [r["split"] for r in rows]
    if first != second or saved != expected or cross_split_pairs(pairs, saved):
        raise ValueError("Saved split determinism or leakage audit failed")
    size = Counter(groups)
    moved = [i for i, (a, b) in enumerate(zip(initial, saved)) if a != b]
    transitions = Counter(f"{a}->{b}" for a, b in zip(initial, saved))
    balances = {}
    for s in SPLITS:
        subset = [r for r in rows if r["split"] == s]
        unanimous = [r for r in subset if r["is_allagree"]]
        balances[s] = {**distribution(subset), "allagree_fraction": len(unanimous)/len(subset),
                       "allagree_distribution": distribution(unanimous),
                       "non_allagree_distribution": distribution([r for r in subset if not r["is_allagree"]])}
    report = {"phase1_manifest_sha256": before, "deterministic_replay": True,
        "saved_assignments_match_replay": True, "rows_reassigned": len(moved),
        "moved_singletons": sum(size[groups[i]] == 1 for i in moved),
        "moved_duplicate_group_rows": sum(size[groups[i]] > 1 for i in moved),
        "transitions": dict(transitions), "initial_cross_split_pairs": len(cross_split_pairs(pairs, initial)),
        "final_cross_split_pairs": len(cross_split_pairs(pairs, saved)),
        "explanation": "When any crossing exists, Phase 1 rebuilds ALL assignments with a seeded greedy group-stratifier. Its objective balances classes, not minimal movement from the initial sklearn split. Singleton groups are reassigned too. This is deterministic, not data loss or 2199 duplicates.",
        "overall": distribution(rows), "balances": balances,
        "limitations": "Class balance is optimized; allagree membership is not a stratification variable. Determinism is verified in the recorded/current software environment; lexical duplicates do not exhaust semantic leakage."}
    out = new_output(output, root)
    write_json(out / "audit.json", report)
    if verify_manifest(root) != before:
        raise RuntimeError("Phase 1 changed during audit")
    return report
