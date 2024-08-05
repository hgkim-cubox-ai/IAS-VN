from .person_dataset import PersonData
from types_ import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


DATASET_DICT = {
    'person_dataset': PersonData,
}

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
            num_workers=cfg['Data']['num_workers'], pin_memory=True,
            sampler=sampler
        )
    
    return dataloader_dict