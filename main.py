import os
import argparse
import yaml

from load import (load_dataloader_dict,
                  load_model_and_optimizer,
                  load_loss_fn_dict)
from train import train
from infer import infer
from utils import setup
from types_ import *


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='experiments/debugging.yaml')
    args = parser.parse_args()
    
    with open(os.path.join(os.getcwd(), args.cfg), 'r') as f:
        cfg = yaml.safe_load(f)
    
    return cfg


def main(cfg):
    # Basic setttings
    rank = setup(cfg)
    # import torch
    # rank = torch.device('cuda:0')
    
    dataloader_dict = load_dataloader_dict(cfg['Data'])
    model, optimizer = load_model_and_optimizer(cfg, rank)
    loss_fn_dict = load_loss_fn_dict(cfg)
    if cfg['mode'] == 'train':
        train(cfg, rank, dataloader_dict, model, optimizer, loss_fn_dict)
    else:
        del dataloader_dict['train']
        infer()


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
    print('Done')
    
    # torchrun --nnodes=1 --nproc_per_node=8 main.py