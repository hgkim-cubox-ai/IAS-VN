import os
import wandb
import torch
from tqdm import tqdm

from utils import (save_checkpoint, send_data_dict_to_device,
                   calculate_accuracy,
                   AverageMeter, AccuracyMeter)


def _train(cfg, rank, loader, model, optimizer, loss_fn_dict, epoch):
    model.train()
    
    total = 0
    correct = 0
    loss_meter = AverageMeter()
    acc_meter = AccuracyMeter()
    loss_fn = loss_fn_dict['ce']['fn']
    pbar = tqdm(loader, desc=f'[Train] Epoch {epoch+1}',
                ncols=150, unit='batch', disable=rank)
    
    for _, input_dict in enumerate(pbar):    
        batch_size = input_dict['input'].size(0)
        input_dict = send_data_dict_to_device(input_dict, rank)
        label = input_dict['label']
        spoof_type = input_dict['spoof_type']
        
        pred = model(input_dict['input'])
        
        if isinstance(loss_fn, torch.nn.BCELoss):
            loss = loss_fn(pred, label.view(-1,1))
        elif isinstance(loss_fn, torch.nn.CrossEntropyLoss):
            loss = loss_fn(pred, spoof_type)
        else:
            loss = None
                                
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred_cls = torch.max(pred.detach(), 1)[1]
        pred_label = torch.where(pred_cls > 0, 0.0, 1.0)
        total += batch_size
        correct += (pred_cls == spoof_type).sum().item()
        
        acc_dict = calculate_accuracy(pred_label, label.view(-1,1),
                                      cfg['threshold'])
        
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc_dict)
                
        pbar.set_postfix(
            loss=loss_meter.avg,
            type_acc=(correct / total * 100),
            label_acc=acc_meter.dict['total']['acc'],
            real=acc_meter.dict['real']['acc'],
            fake=acc_meter.dict['fake']['acc']
        )
    
    ret = []
    ret.append(acc_meter.dict['total']['acc'])
    ret.append(acc_meter.dict['real']['acc'])
    ret.append(acc_meter.dict['fake']['acc'])
    ret.append(correct / total * 100)
    ret.append(loss_meter.avg)
    
    return ret


def _validate(cfg, rank, loader, model, loss_fn_dict, epoch, data_split):
    model.eval()
    
    total = 0
    correct = 0
    loss_meter = AverageMeter()
    acc_meter = AccuracyMeter()
    loss_fn = loss_fn_dict['ce']['fn']
    pbar = tqdm(loader, desc=f'[{data_split}] Epoch {epoch+1}',
                ncols=150, unit='batch', disable=rank)
    
    for i, input_dict in enumerate(pbar):
        batch_size = input_dict['input'].size(0)
        input_dict = send_data_dict_to_device(input_dict, rank)
        label = input_dict['label']
        spoof_type = input_dict['spoof_type']
        
        with torch.no_grad():
            pred = model(input_dict['input'])
        
        if isinstance(loss_fn, torch.nn.BCELoss):
            loss = loss_fn(pred, label.view(-1,1))
        elif isinstance(loss_fn, torch.nn.CrossEntropyLoss):
            loss = loss_fn(pred, spoof_type)
        else:
            loss = None
        
        pred_cls = torch.max(pred.detach(), 1)[1]
        pred_label = torch.where(pred_cls > 0, 0.0, 1.0)
        total += batch_size
        correct += (pred_cls == spoof_type).sum().item()
        
        acc_dict = calculate_accuracy(pred_label, label.view(-1,1),
                                      cfg['threshold'])
        
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc_dict)
                
        pbar.set_postfix(
            loss=loss_meter.avg,
            type_acc=(correct / total * 100),
            label_acc=acc_meter.dict['total']['acc'],
            real=acc_meter.dict['real']['acc'],
            fake=acc_meter.dict['fake']['acc']
        )
    
    ret = []
    ret.append(acc_meter.dict['total']['acc'])
    ret.append(acc_meter.dict['real']['acc'])
    ret.append(acc_meter.dict['fake']['acc'])
    ret.append(correct / total * 100)
    ret.append(loss_meter.avg)
    
    return ret
            

def train(cfg, rank, dataloader_dict, model, optimizer, loss_fn_dict):
    log = []
    max_acc = 0
    
    for epoch in range(cfg['num_epochs']):
        acc_dict = {}
        
        # Train
        acc = _train(cfg, rank, dataloader_dict['train'],
                     model, optimizer, loss_fn_dict, epoch)
        acc_dict['train'] = acc
        log.append(f'Epoch: {epoch+1}\n'
                   f'[Train] real acc: {acc[1]:.3f}\tfake acc: {acc[2]:.3f}\n')
        
        # Val, test
        for data_split in [d for d in dataloader_dict if d != 'train']:
            acc = _validate(cfg, rank, dataloader_dict[data_split],
                            model, loss_fn_dict, epoch, data_split)
            acc_dict[data_split] = acc
            log.append(f'[{data_split}] real acc: {acc[1]:.3f}\tfake acc: {acc[2]:.3f}\n')

        cur_acc = (acc[1] + acc[2]) / 2
        is_best = cur_acc > max_acc
        max_acc = max(cur_acc, max_acc)
        
        if cfg['mode'] == 'train' and rank == 0:
            for data_split, acc in acc_dict.items():
                wandb.log(
                    {
                        f'{data_split} loss': acc[4],
                        f'{data_split} type_acc': acc[3],
                        f'{data_split} label_acc': acc[0],
                        f'{data_split} real': acc[1],
                        f'{data_split} fake': acc[2]
                    },
                    step=epoch+1
                )
                
            save_checkpoint(
                is_best,
                {'epoch': epoch+1, 'state_dict': model.state_dict()},
                cfg['save_path']
            )
    
    log_path = os.path.join(cfg['save_path'], 'log.txt')
    with open(log_path, 'w') as f:
        f.writelines(log)