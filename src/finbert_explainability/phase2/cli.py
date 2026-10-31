import argparse
import json

from .settings import Settings


def main():
    parser = argparse.ArgumentParser(description="Phase 2: frozen-split model evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Read-only replay and balance audit of Phase 1")
    audit.add_argument("--phase1", default="artifacts/phase1")
    audit.add_argument("--output", default="artifacts/phase2/audit")
    smoke = sub.add_parser("smoke", help="Offline synthetic CPU smoke; no pretrained weights downloaded")
    smoke.add_argument("--output", default="artifacts/phase2/smoke-final")
    train = sub.add_parser("train", help="Validation-only search and three final fits; no test evaluation")
    train.add_argument("kind", choices=["baseline", "bert", "finbert"])
    train.add_argument("--config", default="configs/phase2.json")
    train.add_argument("--output", required=True)
    train.add_argument("--approve-training", action="store_true", help="Only use after explicit user approval")
    train.add_argument("--smoke-receipt", default="artifacts/phase2/smoke-final/smoke_receipt.json")
    evaluate = sub.add_parser("evaluate", help="Evaluate a completed, frozen selection run on the saved test split")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--cpu", action="store_true")
    reference = sub.add_parser("reference", help="Non-independent external reference, inference only")
    reference.add_argument("--config", default="configs/phase2.json")
    reference.add_argument("--output", required=True)
    reference.add_argument("--cpu", action="store_true")
    compare = sub.add_parser("compare", help="Combine reports only if test fingerprints match")
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--phase1", default="artifacts/phase1")
    args = parser.parse_args()
    if args.command == "audit":
        from .audit import audit_phase1
        result = audit_phase1(args.phase1, args.output)
        print(json.dumps(result, indent=2))
    elif args.command == "smoke":
        from .smoke import smoke_test
        print(smoke_test(args.output))
    elif args.command == "compare":
        from .workflows import compare_evaluations
        print(compare_evaluations(args.runs, args.output, args.phase1))
    else:
        from .workflows import evaluate_workflow, reference_workflow, train_workflow
        if args.command == "train":
            result = train_workflow(Settings.load(args.config), args.kind, args.output,
                approved=args.approve_training, smoke_receipt=args.smoke_receipt)
        elif args.command == "evaluate":
            result = evaluate_workflow(args.run, args.output, cpu=args.cpu)
        else:
            result = reference_workflow(Settings.load(args.config), args.output, cpu=args.cpu)
        print(result)


if __name__ == "__main__":
    main()
