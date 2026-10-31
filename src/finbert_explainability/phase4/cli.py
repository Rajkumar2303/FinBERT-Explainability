import argparse
from pathlib import Path

from ..phase3.io import read_json
from .pipeline import run


def main():
    parser=argparse.ArgumentParser(description='Phase 4: frozen-model logit LayerIntegratedGradients')
    sub=parser.add_subparsers(dest='command',required=True)
    experiment=sub.add_parser('run',help='Verify inputs and compute selected or full-test IG')
    experiment.add_argument('--config',default='configs/phase4.json')
    experiment.add_argument('--cpu',action='store_true')
    experiment.add_argument('--full-test',action='store_true',help='All frozen test rows; ten fixed visualizations')
    experiment.add_argument('--resume',action='store_true',help='Validate and reuse sealed records in an identical run')
    smoke=sub.add_parser('smoke',help='Tiny randomly initialized CPU BERT; no saved-model changes')
    smoke.add_argument('--output',default='artifacts/phase4/smoke')
    smoke.add_argument('--cuda',action='store_true',help='Run the same tiny execution check on CUDA')
    args=parser.parse_args()
    if args.command=='run': run(read_json(args.config),args.cpu,args.full_test,args.resume)
    else:
        from .smoke import run_smoke
        run_smoke(Path(args.output),cuda=args.cuda)
