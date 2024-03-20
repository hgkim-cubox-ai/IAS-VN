import os
import argparse
import yaml

from types_ import *


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='experiments/debugging.yaml')
    args = parser.parse_args()
    
    with open(os.path.join(os.getcwd(), args.cfg), 'r') as f:
        cfg = yaml.safe_load(f)
    
    return cfg


def main(cfg):
    print('')


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
    print('Done')