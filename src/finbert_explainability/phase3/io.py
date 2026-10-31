import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ..data import LABELS


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Missing or invalid JSON artifact: {path}: {error}") from error


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def digest(path):
    result = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                result.update(chunk)
    except OSError as error:
        raise ValueError(f"Missing/unreadable artifact: {path}") from error
    return result.hexdigest()


def verify_manifest(root, filename):
    root = Path(root).resolve()
    expected = read_json(root / filename)
    if not expected:
        raise ValueError(f"Empty manifest: {root / filename}")
    for name, checksum in expected.items():
        path = (root / name).resolve()
        if root not in path.parents or digest(path) != checksum:
            raise ValueError(f"Manifest mismatch: {path}")
    return expected


def require_listed(root, names, filename="run_manifest.json"):
    manifest = read_json(Path(root) / filename)
    for name in names:
        if name not in manifest:
            raise ValueError(f"Required artifact is not sealed: {root}/{name}")


def snapshot(roots):
    return {str(p.resolve()): digest(p) for root in roots for p in sorted(Path(root).rglob("*")) if p.is_file()}


def protected_output(output, roots, report):
    output, report = Path(output).resolve(), Path(report).resolve()
    for root in map(lambda p: Path(p).resolve(), roots):
        for path in (output, report):
            if path == root or root in path.parents or path in root.parents:
                raise ValueError(f"Phase 1/2 artifacts are read-only: {path}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results: {output}")
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report}")
    output.mkdir(parents=True, exist_ok=True)
    return output, report


def validate_probabilities(probabilities, count):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (count, 3) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite N x 3 values in [0,1]")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Probabilities do not sum to one")
    return p


def load_predictions(path, rows):
    predictions = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if len(predictions) != len(rows) or len({p["row_id"] for p in predictions}) != len(rows):
        raise ValueError(f"Missing or duplicate prediction rows: {path}")
    for p, row in zip(predictions, rows):
        if p["row_id"] != row["row_id"] or p["true_label"] != row["label"] or p["split"] != row["split"] or p["is_allagree"] != row["is_allagree"]:
            raise ValueError(f"Sample alignment/label/subgroup mismatch: {path}: {row['row_id']}")
        if set(p["probabilities"]) != set(LABELS):
            raise ValueError(f"Probability label mapping mismatch: {path}")
    probabilities = validate_probabilities([[p["probabilities"][label] for label in LABELS] for p in predictions], len(rows))
    if not np.array_equal(probabilities.argmax(axis=1), [p["predicted_label"] for p in predictions]):
        raise ValueError(f"Predicted class disagrees with probabilities: {path}")
    return probabilities
