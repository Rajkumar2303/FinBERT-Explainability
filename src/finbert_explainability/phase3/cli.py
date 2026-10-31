import argparse
from pathlib import Path

from .io import read_json
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Read-only Phase 3 reliability analysis; no training")
    parser.add_argument("command", choices=["smoke", "analyze"])
    parser.add_argument("--config", default="configs/phase3.json")
    parser.add_argument("--output", help="New output directory; existing results are protected")
    parser.add_argument("--report", help="New Markdown report path")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for checkpoint inference")
    parser.add_argument("--smoke-receipt", help="Passing current-code smoke receipt required for full analysis")
    args = parser.parse_args()
    cfg = read_json(args.config)
    smoke = args.command == "smoke"
    if smoke:
        cfg["output_dir"] = args.output or "artifacts/phase3-smoke"
        cfg["report_path"] = args.report or str(Path(cfg["output_dir"]) / "SMOKE_RESULTS.md")
    else:
        if args.output:
            cfg["output_dir"] = args.output
        if args.report:
            cfg["report_path"] = args.report
    if args.smoke_receipt:
        cfg["smoke_receipt"] = args.smoke_receipt
    print(run(cfg, cpu=args.cpu or smoke, smoke=smoke))


if __name__ == "__main__":
    main()
