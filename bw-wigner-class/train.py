# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:09:47 2022
TODO:
    Update dataset to encode species as integer and
    store that info somewhere
@author: tnsak
"""
from dataset import BWDataset
from model import BeakerNet
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, RandomAffine, GaussianBlur, Normalize
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn as nn
from tqdm import trange
import torch
import glob
import yaml
import argparse
from util import init_seed, CrossEntSNR, MySelCE
import numpy as np
from torch.optim.lr_scheduler import StepLR
import sklearn.metrics as met

def create_dataloader(cfg, split='train'):
    '''
    create a dataloader for a dataset instance
    '''
    
    label_csv = os.path.join(cfg['label_dir'], cfg['label_csv'][split])
    base_trans = Compose([ToPILImage(), 
                    Resize([224, 224]), 
                    ToTensor(),
                    Normalize(mean=cfg['norm_mean'],
                                        std=cfg['norm_sd'])])
    trans_dict = {
        'train': Compose([ToPILImage(), 
                          Resize([224, 224]),
                          RandomAffine(degrees=0,
                                       translate=(cfg['rndaff_transx'], 0),
                                       fill=cfg['rndaff_fill']),
                          GaussianBlur(kernel_size=(cfg['gb_kernel'],
                                                    cfg['gb_kernel']),
                                       sigma=(0.1, cfg['gb_max'])),
                          ToTensor(),
                          Normalize(mean=cfg['norm_mean'],
                                              std=cfg['norm_sd'])]),
        'val': base_trans,
        'test': base_trans,
        'predict': base_trans
        }
    
    dataset = BWDataset(cfg, label_csv, trans_dict[split])
    if split == 'train':
        if cfg['weighted_sampler']:
            sp = dataset.species
            weights = np.array([len(sp)/sp.count(x) for x in range(cfg['num_classes'])])
            samp_weight= weights[sp]
            sampler = WeightedRandomSampler(samp_weight, len(samp_weight))
            shuffle = False
        else:
            shuffle = True
            sampler = None
    else:
        sampler = None
        shuffle = False
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=shuffle,
        sampler=sampler
        )
    return dataloader

def load_model(cfg):
    '''
    creates a new model or loads existing if one found
    '''
    model = BeakerNet(cfg)
    model_dir = cfg['model_save_dir']
    model_states = glob.glob(model_dir + '/*.pt')

    if cfg['resume'] and len(model_states) > 0:
        # found a save state
        model_epochs = [int(m.replace(model_dir + '/','').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)
        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(model_dir+f'/{start_epoch}.pt', 'rb'), map_location='cpu')
        model.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model, start_epoch

def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    model_dir = cfg['model_save_dir']
    os.makedirs(model_dir, exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(model_dir + f'/{epoch}.pt', 'wb'))
    
    # also save config file if not present
    cfpath = model_dir + '/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)

def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer


def train(cfg, dataloader, model, optimizer):
    '''
    train me models

    Parameters
    ----------
    cfg : TYPE
        DESCRIPTION.
    dataloader : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    optimizer : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    device = cfg['device']
    # send to device and set to train mode
    model.to(device)
    model.train()

    if cfg['weighted_loss']:
        sp = dataloader.dataset.species
        weights = [len(sp)/sp.count(x) for x in range(cfg['num_classes'])]
        weights = torch.FloatTensor(weights).to(device)
    else:
        weights = torch.ones(cfg['num_classes']).to(device)

    # define loss function - MAY CHANGE LATER
    # criterion = nn.CrossEntropyLoss(weight = weights)
    if cfg['do_selnet']:
        sel_criterion = MySelCE(weight=weights,
                                coverage=cfg['sel_coverage'],
                                lam=cfg['sel_lambda'])
        # weights = torch.concat((weights, torch.ones(1).to(device)))
        alpha = cfg['sel_alpha']
        sel_total = 0.0
    criterion = CrossEntSNR(weight=weights)
    # init running averages
    loss_total, oa_total = 0.0, 0.0
    # class_total = torch.zeros(cfg['num_classes']).cpu()
    # class_count = torch.zeros(cfg['num_classes']).cpu()
    all_pred = np.empty(0)
    all_true = np.empty(0)
    pb = trange(len(dataloader))
    has_extras = cfg['use_ici'] + cfg['extra_params']
    
    for idx, (data, label, snr, extras) in enumerate(dataloader):
        # put on device for model speed
        all_true = np.append(all_true, label.cpu().detach().numpy())
        data, label, snr = data.to(device), label.to(device), snr.to(device)
        # forward, beakernet!
        if has_extras:
            extras = extras.to(device)
        else:
            extras = None
        if cfg['do_selnet']:
            prediction, sel_pred = model(data, extras)
        else:
            prediction = model(data, extras)
        # have to reste grads
        optimizer.zero_grad()
        # calc loss and full send back
        loss = criterion(prediction, label, snr)
        if cfg['do_selnet']:
            sel_loss = sel_criterion(sel_pred, label)
            loss = loss * alpha + (1-alpha) * sel_loss
            sel_total += sel_loss.item()
            
        loss.backward()
        
        optimizer.step()
    
        loss_total += loss.item()
        
        pred_label = torch.argmax(prediction, dim=1)
        all_pred = np.append(all_pred, pred_label.cpu().detach().int().numpy())
        oa = torch.mean((pred_label == label).float())
        oa_total += oa.item()
        # for s in range(len(class_total)):
        #     class_count[s] += torch.sum(label == s).cpu()
        #     class_total[s] += torch.sum(pred_label[label == s] == s).cpu()
         
        # cba = [(x / max(y, 1)) for x, y in zip(class_total, class_count)]
        
        pb.set_description(
                '[Train] Loss {:.2f}; OA {:.2f}%; Sel {:.2f}'.format(
                loss_total/(idx + 1),
                100*oa_total/(idx + 1),
                sel_total/(idx+1)
                # 100*np.mean(cba)
                )
            )
        pb.update(1)
    
    pb.close()
    loss_total /= len(dataloader)
    oa_total /= len(dataloader)
    
    return(loss_total, oa_total, all_pred, all_true)

def validate(cfg, dataloader, model):
    device = cfg['device']
    model = model.to(device)
    model.eval()

    if cfg['weighted_loss']:
        sp = dataloader.dataset.species
        weights = [len(sp)/sp.count(x) for x in range(cfg['num_classes'])]
        weights = torch.FloatTensor(weights).to(device)
    else:
        weights = torch.ones(cfg['num_classes']).to(device)

    # criterion = nn.CrossEntropyLoss(weight=weights)
    if cfg['do_selnet']:
        sel_criterion = MySelCE(weight=weights,
                                coverage=cfg['sel_coverage'],
                                lam=cfg['sel_lambda'])
        # weights = torch.concat((weights, torch.ones(1).to(device)))
        alpha = cfg['sel_alpha']
        
    criterion = CrossEntSNR(weight=weights)
    loss_total, oa_total = 0.0, 0.0
    # class_total = torch.zeros(cfg['num_classes']).cpu()
    # class_count = torch.zeros(cfg['num_classes']).cpu()
    all_pred = np.empty(0)
    all_true = np.empty(0)
    pb = trange(len(dataloader))
    has_extras = cfg['use_ici'] + cfg['extra_params']
    # this is so we dont calc gradient bc not needed for val
    with torch.no_grad():
        for idx, (data, label, snr, extras) in enumerate(dataloader):
            all_true = np.append(all_true, label.cpu().numpy())
            data, label, snr = data.to(device), label.to(device), snr.to(device)
            if has_extras:
                extras = extras.to(device)
            else:
                extras = None
            if cfg['do_selnet']:
                prediction, sel_pred = model(data, extras)
            else:
                prediction = model(data, extras)
            loss = criterion(prediction, label, snr)
            
            loss_total += loss.item()
            
            pred_label = torch.argmax(prediction, dim=1)
            all_pred = np.append(all_pred, pred_label.cpu().int().numpy())
            # pred_score = torch.softmax(prediction, dim=1)
            oa = torch.mean((pred_label == label).float())
            # for s in range(len(class_total)):
            #     class_count[s] += torch.sum(label == s).cpu()
            #     class_total[s] += torch.sum(pred_label[label == s] == s).cpu()
            # # do this way cuz no div0
            # cba = [(x / max(y, 1)) for x, y in zip(class_total, class_count)]
            oa_total += oa.item()
            
            pb.set_description(
                '[Val] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total / (idx + 1),
                    100 * oa_total / (idx + 1)
                    # 100 * np.mean(cba)
                    )
                )
            pb.update(1)
    
    pb.close()
    loss_total /= len(dataloader)
    oa_total /= len(dataloader)
    
    return(loss_total, oa_total, all_pred, all_true)
    
def main():
    # set up command line argument parser for cfg file
    parser = argparse.ArgumentParser(description='Train yo BeakerNet CLICK CLICK BOIIII')
    parser.add_argument('--config', help='Path to config file', default='configs/bn1_resnet50.yaml')

    args = parser.parse_args()
    
    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    
    # init random number generator seed (set at the start)
    # (note this tries to get from dict and has fail default)
    init_seed(cfg.get('seed', None))
    
    device = cfg['device']
    # adding stuff that might be m
    if not 'do_selnet' in cfg.keys():
        cfg['do_selnet'] = False
    if not 'model' in cfg.keys():
        cfg['model'] = 'r18'
        
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'


    
       # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')
            
    # initialize model
    model, current_epoch = load_model(cfg)
    # just jank to write on example model
    save_model(cfg, current_epoch, model, stats={
        'loss_train': 0.0,
        'loss_val': 0.0,
        'oa_train': 0.0,
        'oa_val': 0.0,
        'cba_train': 0.0,
        'cba_val': 0.0
    })
    # set up model optimizer
    optim = setup_optimizer(cfg, model)
    
    writer = SummaryWriter()
    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    scheduler = StepLR(optim, step_size=cfg['lr_step_count'], gamma=cfg['lr_step'])
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train, pred_train, true_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val, pred_val, true_val = validate(cfg, dl_val, model)
        scheduler.step()
        cr_train = met.classification_report(true_train, pred_train, output_dict=True)
        cr_val = met.classification_report(true_val, pred_val, output_dict=True)
        t_rep = {y: 
                 {str(x): cr_train[str(x)][y] for x in np.unique(true_train)}
                 for y in ['precision','recall']}
        v_rep = {y: 
                 {str(x): cr_val[str(x)][y] for x in np.unique(true_val)}
                 for y in ['precision','recall']}  
        writer.add_scalars('Train/prec', t_rep['precision'], current_epoch)
        writer.add_scalars('Val/prec', v_rep['precision'], current_epoch)
        writer.add_scalars('Train/recall', t_rep['recall'], current_epoch)
        writer.add_scalars('Val/recall', v_rep['recall'], current_epoch)
        writer.add_scalar('Train/AvgPrec', cr_train['macro avg']['precision'], current_epoch)
        writer.add_scalar('Val/AvgPrec', cr_val['macro avg']['precision'], current_epoch)
        writer.add_scalar('Train/OA', oa_train, current_epoch)
        writer.add_scalar('Train/Loss', loss_train, current_epoch)
        writer.add_scalar('Train/AvgRcl', cr_train['macro avg']['recall'], current_epoch)
        writer.add_scalar('Train/AvgF1', cr_train['macro avg']['f1-score'], current_epoch)
        writer.add_scalar('Val/OA', oa_val, current_epoch)
        writer.add_scalar('Val/Loss', loss_val, current_epoch)
        writer.add_scalar('Val/AvgRcl', cr_val['macro avg']['recall'], current_epoch)
        writer.add_scalar('Val/AvgF1', cr_val['macro avg']['f1-score'], current_epoch)
        # combine stats and save
        stats = {
           'loss_train': loss_train,
           'loss_val': loss_val,
           'oa_train': oa_train,
           'oa_val': oa_val
            # 'cba_train': np.mean(cba_train),
            # 'cba_val': np.mean(cba_val)
        }
        writer.flush()
        save_model(cfg, current_epoch, model, stats)
    writer.close()
   # WE DID IT BEAKERNET ONLINE
    
if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
