import argparse
import yaml

from load import (load_dataloader_dict,
                  load_model_and_optimizer,
                  load_loss_fn_dict)
from train import train
from utils import setup
from types_ import *


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='experiments/debugging/debugging.yaml')
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg['cfg'] = args.cfg
    
    return cfg


def main(cfg):
    # Basic setttings
    rank = setup(cfg)
    
    dataloader_dict = load_dataloader_dict(cfg)
    model, optimizer = load_model_and_optimizer(cfg, rank)
    loss_fn_dict = load_loss_fn_dict(cfg)
    
    train(cfg, rank, dataloader_dict, model, optimizer, loss_fn_dict)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
    print('Done')
    
    """
    if debigging, python main.py
    if training, torchrun --nnodes=1 --nproc_per_node=4 main.py
    """