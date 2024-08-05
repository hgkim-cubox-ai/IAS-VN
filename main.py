import argparse

import yaml

from data import load_dataloader_dict
from load import load_loss_fn_dict, load_model_and_optimizer
from train import train
from types_ import *
from utils import setup


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", "-C", type=str, default="experiments/debugging/debugging.yaml"
    )
    parser.add_argument("--num_workers", "-W", type=int)
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
        cfg["cfg"] = args.cfg
        if args.num_workers is not None:
            cfg["Data"]["num_workers"] = args.num_workers

    return cfg


def main():
    cfg = parse_args()

    # Basic setttings
    rank = setup(cfg)

    dataloader_dict = load_dataloader_dict(cfg)
    model, optimizer = load_model_and_optimizer(cfg, rank)
    loss_fn_dict = load_loss_fn_dict(cfg)

    train(cfg, rank, dataloader_dict, model, optimizer, loss_fn_dict)


if __name__ == "__main__":
    main()

    """
    if debugging
        python main.py
    if training
        torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg path/to/yaml
    """
