import os
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.distributed as dist

from types_ import *


def set_seed(seed: int = 777) -> None:
    """
    Set the seed for reproducible result
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_path(path: str) -> None:
    """
    Make directory if not exist.
    Else, raise error.

    Args:
        path (str): path to directory
    """
    try:
        os.mkdir(path)
    except:
        raise ValueError(f'Existing path!: {path}')


def setup(cfg: Dict[str, Any]) -> int:
    """
    Basic settings for training. (default to multi-gpu)

    Args:
        cfg (Dict[str, Any]): config as dictionary.

    Returns:
        int: rank
    """
    dist.init_process_group(cfg['distributed']['backend'])
    rank = dist.get_rank()
    
    seed = cfg['seed'] * dist.get_world_size() + rank
    set_seed(seed)
    
    if rank == 0:
        prepare_path(cfg['save_path'])
    
    return rank


def send_data_dict_to_device(
    data: Dict[str, Any],
    rank: int
) -> Dict[str, Any]:
    """
    Send data from cpu to gpu.

    Args:
        data (Dict[str, Any]): data dictionary from data loader
        rank (int): cpu or cuda or rank

    Returns:
        Dict[str, Any]: data dictionary on rank
    """
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(rank)
        elif isinstance(v, list):
            data_list = []
            for i in range(len(v)):
                data_list.append(v[i].detach().to(rank))
            data[k] = data_list
    
    return data


def is_image_file(filename: str) -> bool:
    """
    Check the input is image file

    Args:
        filename (str): path to file

    Returns:
        bool: True or False
    """
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
                      '.ppm', '.PPM', '.bmp', '.BMP']
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)