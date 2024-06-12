import os
import random
import shutil
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


def calculate_accuracy(pred, label, threshold=0.5):
    # pred = torch.sigmoid(pred)
    pred = torch.where(pred > threshold, 1.0, 0.0)
    acc = (pred == label).float().mean()
    # acc = (pred.round() == label).float().mean()
    
    pred = pred.view(-1)
    label = label.view(-1)
        
    real_indices = (label == 1).nonzero().view(-1)
    fake_indices = (label == 0).nonzero().view(-1)
    
    real_acc = (pred[real_indices] == label[real_indices]).float().sum()
    fake_acc = (pred[fake_indices] == label[fake_indices]).float().sum()
    # real_acc = (pred[real_indices].round() == label[real_indices]).float().sum()
    # fake_acc = (pred[fake_indices].round() == label[fake_indices]).float().sum()
    
    acc.mul_(100)
    real_acc.div_(len(real_indices) + 1e-8).mul_(100)
    fake_acc.div_(len(fake_indices) + 1e-8).mul_(100)
        
    return acc, real_acc, fake_acc


def save_checkpoint(is_best, state, save_path):    
    save_path = os.path.join(save_path, 'saved_models')
    os.makedirs(save_path, exist_ok=True)
    filename = '%d.pth' % (state['epoch'])
    best_filename = 'best_' + filename
    file_path = os.path.join(save_path, filename)
    best_file_path = os.path.join(save_path, best_filename)
    torch.save(state, file_path)
    
    # Remove previous best model
    if is_best:
        saved_models = os.listdir(save_path)
        for saved_model in saved_models:
            if saved_model.startswith('best'):
                os.remove(os.path.join(save_path, saved_model))
        shutil.copyfile(file_path, best_file_path)



class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total = 0
        self.correct = 0
        self.acc = 0
    
    def update(self, t, c):
        self.total += t
        self.correct += c
        self.acc = self.correct / self.total * 100