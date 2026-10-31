import argparse

parser=argparse.ArgumentParser(description='Phase 6 saved-artifact audit; no training or inference')
parser.add_argument('command',choices=['audit'])
parser.add_argument('--output',default='artifacts/phase6/audit')
args=parser.parse_args()
from .audit import run
run(args.output)
