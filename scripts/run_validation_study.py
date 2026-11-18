"""Run/resume M10. All stages execute sequentially to isolate GPU timing."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prismserve.validation_study import run_study,STAGES
from prismserve.study_monitor import machine_conditions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/validation_study_ac.json'))
    parser.add_argument('--output',type=Path,default=Path('results/validation_study_final'))
    parser.add_argument('--stage',choices=['all',*STAGES],default='all')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config.get('required_ac_power') and machine_conditions()['power_plugged'] is not True:
        raise RuntimeError('This configuration requires AC power. Plug in the laptop before running; existing results are preserved.')
    print(json.dumps(run_study(args.config,args.output,args.stage),indent=2))


if __name__=='__main__':
    main()
