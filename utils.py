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
        int: rank in ddp or 0
    """
    if cfg['mode'] == 'debugging':
        device = 0
        seed = cfg['seed']
    else:   # train
        dist.init_process_group(cfg['distributed']['backend'])
        device = dist.get_rank()
        seed = cfg['seed'] * dist.get_world_size() + device
    set_seed(seed)
    
    if device == 0:
        os.makedirs(cfg['save_path'], exist_ok=True)
        shutil.copy(cfg['cfg'], os.path.join(cfg['save_path'],cfg['cfg'].split('/')[1]))
    
    return device


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


def load_state_dict(path):
    state_dict = OrderedDict()
    tmp = torch.load(path)['state_dict']
    for n, v in tmp.items():
        if n.startswith('module'):
            state_dict[n[7:]] = v
        else:
            state_dict[n] = v
    return state_dict


def calculate_accuracy(pred, label, th=0.5):
    acc_dict = {'total': {}, 'real':{}, 'fake': {}}
    
    pred = torch.where(pred > th, 1.0, 0.0).view(-1)
    label = label.view(-1)
    
    idx_r = (label==1).nonzero().view(-1)
    idx_f = (label==0).nonzero().view(-1)
    num_correct_r = (pred[idx_r]==label[idx_r]).float().sum().item()
    num_correct_f = (pred[idx_f]==label[idx_f]).float().sum().item()
    
    acc_dict['real']['num'] = len(idx_r)
    acc_dict['real']['correct'] = num_correct_r
    acc_dict['fake']['num'] = len(idx_f)
    acc_dict['fake']['correct'] = num_correct_f
    acc_dict['total']['num'] = len(idx_r) + len(idx_f)
    acc_dict['total']['correct'] = num_correct_r + num_correct_f
    
    return acc_dict


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
        acc_dict = {'num': 1e-8, 'correct': 0, 'acc': 0}
        self.dict = {
            'total': acc_dict.copy(),
            'real': acc_dict.copy(),
            'fake': acc_dict.copy()
        }
    
    def update(self, acc_dict):
        for k, subdict in acc_dict.items():
            for n, v in subdict.items():
                self.dict[k][n] += v
            self.dict[k]['acc'] = (self.dict[k]['correct']/self.dict[k]['num'])*100