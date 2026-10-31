import argparse
from dataclasses import replace
import json

from .config import Config
from .pipeline import prepare


def main():
    parser = argparse.ArgumentParser(description="Prepare Financial PhraseBank; no training")
    parser.add_argument("--config", default="configs/phase1.json")
    parser.add_argument("--output", help="Override output directory (must be empty)")
    parser.add_argument("--archive", help="Use a local, checksum-verified source ZIP offline")
    args = parser.parse_args()
    config = Config.load(args.config)
    if args.output:
        config = replace(config, output_dir=args.output)
    stats = prepare(config, args.archive)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
