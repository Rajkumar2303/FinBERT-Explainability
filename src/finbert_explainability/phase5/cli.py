import argparse

from ..phase3.io import read_json


def main():
    parser=argparse.ArgumentParser(description='Phase 5: frozen FinBERT IG versus attention')
    sub=parser.add_subparsers(dest='command',required=True)
    run_parser=sub.add_parser('run',help='Run CUDA saved-checkpoint experiment')
    run_parser.add_argument('--config',default='configs/phase5.json')
    run_parser.add_argument('--resume',action='store_true')
    run_parser.add_argument('--smoke',action='store_true',help='Ten fixed Phase 4 examples on the actual CUDA checkpoint')
    smoke_parser=sub.add_parser('cpu-smoke',help='Ten synthetic sentences on tiny CPU BERT')
    smoke_parser.add_argument('--config',default='configs/phase5.json')
    smoke_parser.add_argument('--output',default='artifacts/phase5/cpu-smoke')
    args=parser.parse_args()
    if args.command=='run':
        from .pipeline import run
        run(read_json(args.config),args.resume,args.smoke)
    else:
        from .smoke import run_smoke
        run_smoke(args.output,read_json(args.config))
