"""Execute/resume the final predeclared capacity study, sequentially."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prismserve.frontier import run_frontier


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/capacity_frontier.json'))
    p.add_argument('--output',type=Path,default=Path('results/capacity_frontier'))
    p.add_argument('--stage',choices=['all','serving','quality'],default='all')
    args=p.parse_args()
    print(json.dumps(run_frontier(args.config,args.output,args.stage),indent=2))


if __name__=='__main__':
    main()
