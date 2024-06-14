import os
import torch
from tqdm import tqdm

from utils import (save_checkpoint, send_data_dict_to_device,
                   calculate_accuracy, AverageMeter,
                   AccuracyMeter)


def _train(cfg, device, loader, model, optimizer, loss_fn_dict, epoch):
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AccuracyMeter()
    loss_fn = loss_fn_dict['bce']['fn']
    pbar = tqdm(loader, desc=f'[Train] Epoch {epoch+1}',
                ncols=150, unit='batch', disable=device)
    
    for _, input_dict in enumerate(pbar):    
        batch_size = input_dict['input'].size(0)
        input_dict = send_data_dict_to_device(input_dict, device)
        
        pred = model(input_dict['input'])
        label = input_dict['label']
        
        loss = loss_fn(pred, label.view(-1,1))
                                
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc_dict = calculate_accuracy(pred.detach(), label.view(-1,1),
                                      cfg['threshold'])
        
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc_dict)
                
        pbar.set_postfix(
            loss=loss_meter.avg,
            acc=acc_meter.dict['total']['acc'],
            real=acc_meter.dict['real']['acc'],
            fake=acc_meter.dict['fake']['acc']
        )
    
    return acc_meter.dict['real']['acc'], acc_meter.dict['fake']['acc']


def _validate(cfg, device, loader, model, loss_fn_dict, epoch, data_split):
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AccuracyMeter()
    loss_fn = loss_fn_dict['bce']['fn']
    pbar = tqdm(loader, desc=f'[{data_split}] Epoch {epoch+1}',
                ncols=150, unit='batch', disable=device)
    
    for i, input_dict in enumerate(pbar):
        batch_size = input_dict['input'].size(0)
        input_dict = send_data_dict_to_device(input_dict, device)
        
        with torch.no_grad():
            pred = model(input_dict['input'])
        label = input_dict['label']
        
        loss = loss_fn(pred, label.view(-1,1))
                            
        acc_dict = calculate_accuracy(pred.detach(), label.view(-1,1),
                                      cfg['threshold'])
        
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc_dict)
                
        pbar.set_postfix(
            loss=loss_meter.avg,
            acc=acc_meter.dict['total']['acc'],
            real=acc_meter.dict['real']['acc'],
            fake=acc_meter.dict['fake']['acc']
        )
    
    return acc_meter.dict['real']['acc'], acc_meter.dict['fake']['acc']
            

def train(cfg, device, dataloader_dict, model, optimizer, loss_fn_dict):
    log = []
    max_acc = 0
    
    for epoch in range(cfg['num_epochs']):
        # Train
        acc = _train(cfg, device, dataloader_dict['train'],
                     model, optimizer, loss_fn_dict, epoch)
        log.append(f'Epoch: {epoch+1}\n'
                   f'[Train] real acc: {acc[0]:.3f}\tfake acc: {acc[1]:.3f}\n')
        
        # Val, test
        for data_split in [d for d in dataloader_dict if d != 'train']:
            acc = _validate(cfg, device, dataloader_dict[data_split],
                            model, loss_fn_dict, epoch, data_split)
            log.append(f'[{data_split}] real acc: {acc[0]:.3f}\tfake acc: {acc[1]:.3f}\n')

        cur_acc = (acc[0] + acc[1]) / 2
        is_best = cur_acc > max_acc
        max_acc = max(cur_acc, max_acc)
        if device == 0:
            save_checkpoint(
                is_best,
                {'epoch': epoch+1, 'state_dict': model.state_dict()},
                cfg['save_path']
            )
    
    log_path = os.path.join(cfg['save_path'], 'log.txt')
    with open(log_path, 'w') as f:
        f.writelines(log)