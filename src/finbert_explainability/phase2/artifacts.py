import hashlib
import json
from pathlib import Path

from ..data import LABELS
from ..splitting import SPLITS


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    for required in [*(f"{s}.jsonl" for s in SPLITS), "config.json", "metadata.json", "leakage_report.json", "initial_assignments.json", "statistics.json"]:
        if required not in manifest:
            raise ValueError(f"Missing manifest entry: {required}")
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Phase 1 integrity check failed: {name}")
    metadata = read_json(root / "metadata.json")
    if metadata["label_mapping"] != {str(i): label for i, label in enumerate(LABELS)}:
        raise ValueError("Unexpected Phase 1 label mapping")
    return digest(root / "manifest.json")


def load_split(root, name):
    if name not in SPLITS:
        raise ValueError(f"Unknown split {name}")
    rows = [json.loads(line) for line in (Path(root) / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    if not rows or any(r["split"] != name or r["label"] not in range(3) for r in rows):
        raise ValueError(f"Invalid {name} split")
    return rows


def new_output(path, phase1):
    out, protected = Path(path).resolve(), Path(phase1).resolve()
    if out == protected or protected in out.parents or out in protected.parents:
        raise ValueError("Phase 1 and its ancestors are read-only for Phase 2 outputs")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def seal(root):
    from ..pipeline import write_json
    root = Path(root)
    write_json(root / "run_manifest.json", {p.relative_to(root).as_posix(): digest(p)
               for p in sorted(root.rglob("*")) if p.is_file() and p.name != "run_manifest.json"})


def verify_run(root):
    root = Path(root).resolve()
    for name, expected in read_json(root / "run_manifest.json").items():
        path = (root / name).resolve()
        if root not in path.parents or digest(path) != expected:
            raise ValueError(f"Experiment integrity check failed: {name}")
