import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from types_ import *
from data import PersonData
from models import MODEL_DICT
from losses import LOSS_FN_DICT
from utils import load_state_dict


def load_dataloader_dict(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Return data dictionary.
    data_split: train/val/test from config

    Args:
        cfg (Dict): config as dictionary

    Returns:
        Dict: {'train': train_loader, 'test': test_loader, 'val': val_loader}
    """
    dataloader_dict = {}
    all_datasets = cfg['Data']['datasets']
    for data_split, dataset_names in all_datasets.items():
        # If test (or val) set is not used.
        if dataset_names is None:
            continue
        
        dataset = PersonData(cfg['Data'], data_split=='train')
        if cfg['mode'] == 'train' and data_split == 'train':
            sampler = DistributedSampler(dataset)
        else:
            sampler = None
        
        dataloader_dict[data_split] = DataLoader(
            dataset=dataset, batch_size=cfg['Data']['batch_size'],
            num_workers=cfg['num_workers'], pin_memory=True,
            sampler=sampler
        )
    
    return dataloader_dict


def load_model_and_optimizer(
    cfg: Dict[str, Any], rank: Union[int, str]
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Args:
        rank (int): gpu id

    Returns:
       model, optimizer.
    """
    # Model
    model = MODEL_DICT[cfg['model']](cfg)
    model.to(rank)
    print(f'Model on device {rank}')
    
    # Load weight
    if cfg['pretrained_model'] is not None:
        path = cfg['pretrained_model']
        state_dict = load_state_dict(os.path.join(os.getcwd(), path))
        model.load_state_dict(state_dict)
        print(f'Load weights from {path}')
    
    if cfg['mode'] == 'train':
        # rank = rank % torch.cuda.device_count()
        model = DDP(model, device_ids=[rank])
    
    # Optimizer
    optimizer = None
    params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            params=params, lr=cfg['base_lr'], weight_decay=cfg['weight_decay']
        )
    elif cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            params=params, lr=cfg['base_lr'], weight_decay=cfg['weight_decay']
        )
    
    return model, optimizer


def load_loss_fn_dict(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    loss_fn_dict = {
        'name': {
            'fn': MSE()
            'weight': 1.0
        }
    }
    Returns:
        Dict[str, Dict[str, Any]]: loss function dict
    """
    loss_fn_dict = {}
    for loss_fn, weight in cfg['loss_functions'].items():
        loss_fn_dict[loss_fn] = {
            'fn': LOSS_FN_DICT[loss_fn](),
            'weight': weight
        }
    
    return loss_fn_dict