"""Measure an existing Phase 2 CLI command without changing its implementation.

Usage: python scripts/run_measured.py RECORD.json train bert [existing CLI flags]
Telemetry is stored outside sealed run directories. No experiment settings change.
"""
import json
from pathlib import Path
import sys
import time
import traceback

import torch

from finbert_explainability.phase2 import neural
from finbert_explainability.phase2.cli import main
from finbert_explainability.phase2.artifacts import digest, verify_manifest


def memory():
    if not torch.cuda.is_available():
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    torch.cuda.synchronize()
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


if __name__ == "__main__":
    record_path = Path(sys.argv[1])
    if record_path.exists():
        raise FileExistsError(record_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    command = sys.argv[2:]
    before = verify_manifest("artifacts/phase1")
    record = {"command": command, "interpreter": sys.executable,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(0), "phase1_manifest_sha256": before,
              "test_sha256": digest("artifacts/phase1/test.jsonl"), "fits": [],
              "memory_definition": "PyTorch peak allocated/reserved device bytes, excluding other processes/driver memory",
              "status": "running", "oom_adjustments": []}

    def save():
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    original = neural.train_once

    def measured_fit(*args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        # Positional signature: train, validation, cfg, model_spec, params, seed, path.
        entry = {"directory": str(args[6]), "seed": args[5], "params": args[4]}
        try:
            result = original(*args, **kwargs)
            entry["status"] = "completed"
            return result
        except BaseException as error:
            entry.update(status="failed", error=repr(error))
            raise
        finally:
            entry.update(wall_seconds=time.perf_counter()-start, **memory())
            record["fits"].append(entry)
            save()

    neural.train_once = measured_fit
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    save()
    try:
        sys.argv = ["finbert-experiment", *command]
        main()
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - start
        peaks = [memory(), *record["fits"]]
        record.update({key: max(p[key] for p in peaks) for key in ("peak_allocated_bytes", "peak_reserved_bytes")})
        record["phase1_unchanged"] = verify_manifest("artifacts/phase1") == before
        save()
