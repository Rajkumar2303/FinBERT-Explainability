from dataclasses import replace
import hashlib
import json
from zipfile import ZipFile

import pytest

from finbert_explainability.config import Config
from finbert_explainability.data import annotate_membership, normalize, read_subset
from finbert_explainability.leakage import connected_groups, cross_split_pairs, detect_duplicates
from finbert_explainability.pipeline import prepare
from finbert_explainability.splitting import grouped_stratified_split, stratified_split


def test_normalization_and_membership():
    assert normalize("  PROFIT\tＲＯＳＥ ") == "profit rose"
    rows = [{"sentence": "Profit rose", "label": 2}, {"sentence": "Profit rose", "label": 0}]
    result = annotate_membership(rows, rows[:1])
    assert [r["is_allagree"] for r in result] == [True, False]
    with pytest.raises(ValueError, match="absent"):
        annotate_membership(rows, [{"sentence": "missing", "label": 0}])


@pytest.mark.parametrize("kwargs", [{"fractions": [0.7, 0.2, 0.2]}, {"fractions": [1, 0, 0]},
    {"near_duplicate_threshold": 0}, {"near_duplicate_threshold": 1.1}, {"seed": -1}, {"revision": "main"}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_duplicate_detection_and_transitivity():
    rows = [{"sentence": text} for text in [
        "The company's operating profit increased significantly during the financial year.",
        " THE COMPANY'S operating profit increased significantly during the financial year. ",
        "The company's operating profit increased significantly during the financial year!",
        "An unrelated acquisition was cancelled."]]
    pairs = detect_duplicates(rows, 0.85)
    assert any(p["kind"] == "exact" and p["left"] == 0 and p["right"] == 1 for p in pairs)
    assert any(p["kind"] == "near" and p["right"] == 2 for p in pairs)
    assert all(p["right"] != 3 for p in pairs)
    groups = connected_groups(4, [{"left": 0, "right": 1}, {"left": 1, "right": 2}])
    assert groups[0] == groups[2] != groups[3]
    assert len(cross_split_pairs(pairs, ["train", "test", "train", "test"])) >= 1


def test_stratification_and_group_integrity():
    rows = [{"label": i % 3} for i in range(300)]
    assignments = stratified_split(rows, (0.7, 0.15, 0.15), 42)
    assert [assignments.count(s) for s in ("train", "validation", "test")] == [210, 45, 45]
    groups = [i // 3 for i in range(300)]
    grouped = grouped_stratified_split(rows, groups, (0.7, 0.15, 0.15), 42)
    assert grouped == grouped_stratified_split(rows, groups, (0.7, 0.15, 0.15), 42)
    for start in range(0, 300, 3):
        assert len(set(grouped[start:start+3])) == 1
    for split, expected in [("train", 210), ("validation", 45), ("test", 45)]:
        assert abs(grouped.count(split) - expected) <= 3
    with pytest.raises(ValueError, match="nonempty"):
        grouped_stratified_split(rows, [0]*300, (0.7, 0.15, 0.15), 42)


def test_pipeline_offline_reproducible(tmp_path):
    archive = tmp_path / "fixture.zip"
    labels = ["negative", "neutral", "positive"]
    lines = [f"Company {i} reported outcome {i*i+11} for product {i*137}.@{labels[i%3]}" for i in range(90)]
    lines += lines[:3]
    with ZipFile(archive, "w") as z:
        z.writestr("FinancialPhraseBank-v1.0/Sentences_50Agree.txt", "\n".join(lines))
        z.writestr("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt", "\n".join(lines[:12]))
    assert len(read_subset(archive, "Sentences_50Agree.txt")) == 93
    cfg = Config(output_dir=str(tmp_path / "first"), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), near_duplicate_threshold=0.99)
    stats = prepare(cfg, archive)
    prepare(replace(cfg, output_dir=str(tmp_path / "second")), archive)
    assert stats["overall"]["rows"] == 93
    assert stats["overall"]["allagree_rows"] == 15
    report = json.loads((tmp_path / "first/leakage_report.json").read_text())
    assert report["exact_pairs"] == 3
    assert report["final_cross_split_pairs"] == []
    ids = []
    for split in ("train", "validation", "test"):
        a = (tmp_path / f"first/{split}.jsonl").read_bytes()
        assert a == (tmp_path / f"second/{split}.jsonl").read_bytes()
        ids.extend(json.loads(line)["row_id"] for line in a.splitlines())
    assert len(ids) == len(set(ids)) == 93
    manifest = json.loads((tmp_path / "first/manifest.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((tmp_path / "first" / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        prepare(cfg, archive)
    with pytest.raises(ValueError, match="SHA256"):
        prepare(replace(cfg, output_dir=str(tmp_path / "bad"), archive_sha256="wrong"), archive)
