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
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, RandomAffine, GaussianBlur
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn as nn
from tqdm import trange
import torch
import glob
import yaml
import argparse
from util import init_seed, CrossEntSNR
import numpy as np
from torch.optim.lr_scheduler import StepLR
import sklearn.metrics as met

def create_dataloader(cfg, split='train'):
    '''
    create a dataloader for a dataset instance
    '''
    label_csv = os.path.join(cfg['label_dir'], cfg['label_csv'][split])
    trans_dict = {
        'train': Compose([ToPILImage(), 
                          Resize([224, 224]),
                          RandomAffine(degrees=0,
                                       translate=(cfg['rndaff_transx'], 0),
                                       fill=cfg['rndaff_fill']),
                          GaussianBlur(kernel_size=(cfg['gb_kernel'],
                                                    cfg['gb_kernel']),
                                       sigma=(0.1, cfg['gb_max'])),
                          ToTensor()]),
        'val': Compose([ToPILImage(), Resize([224, 224]), ToTensor()]),
        'test': Compose([ToPILImage(), Resize([224, 224]), ToTensor()]),
        'predict': Compose([ToPILImage(), Resize([224, 224]), ToTensor()])
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
    if (cfg['resume'] & len(model_states)):
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
        sp = dataloader.dataset.species.cpu()
        weights = [len(sp)/sp.count(x) for x in range(cfg['num_classes'])]
        weights = torch.FloatTensor(weights).to(device)
    else:
        weights = torch.ones(cfg['num_classes']).to(device)
    # define loss function - MAY CHANGE LATER
    # criterion = nn.CrossEntropyLoss(weight = weights)
    criterion = CrossEntSNR(weight=weights)
    # init running averages
    loss_total, oa_total = 0.0, 0.0
    # class_total = torch.zeros(cfg['num_classes']).cpu()
    # class_count = torch.zeros(cfg['num_classes']).cpu()
    all_pred = np.empty(0).cpu()
    pb = trange(len(dataloader))
    
    for idx, (data, label, snr, extras) in enumerate(dataloader):
        # put on device for model speed
        data, label, snr = data.to(device), label.to(device), snr.to(device)
        # forward, beakernet!
        if cfg['extra_params']:
            extras = extras.to(device)
        else:
            extras = None
        prediction = model(data, extras)
        # have to reste grads
        optimizer.zero_grad()
        # calc loss and full send back
        loss = criterion(prediction, label, snr)
        loss.backward()
        
        optimizer.step()
    
        loss_total += loss.item()
        
        pred_label = torch.argmax(prediction, dim=1)
        all_pred = np.append(all_pred, pred_label.cpu().detach().numpy())
        oa = torch.mean((pred_label == label).float())
        oa_total += oa.item()
        # for s in range(len(class_total)):
        #     class_count[s] += torch.sum(label == s).cpu()
        #     class_total[s] += torch.sum(pred_label[label == s] == s).cpu()
         
        # cba = [(x / max(y, 1)) for x, y in zip(class_total, class_count)]
        
        pb.set_description(
                '[Train] Loss {:.2f}; OA {:.2f}%;'.format(
                loss_total/(idx + 1),
                100*oa_total/(idx + 1)
                # 100*np.mean(cba)
                )
            )
        pb.update(1)
    
    pb.close()
    loss_total /= len(dataloader)
    oa_total /= len(dataloader)
    
    return(loss_total, oa_total, all_pred)

def validate(cfg, dataloader, model):
    device = cfg['device']
    model = model.to(device)
    model.eval()
    if cfg['weighted_loss']:
        sp = dataloader.dataset.species.cpu()
        weights = [len(sp)/sp.count(x) for x in range(cfg['num_classes'])]
        weights = torch.FloatTensor(weights).to(device)
    else:
        weights = torch.ones(cfg['num_classes']).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = CrossEntSNR(weight=weights)
    loss_total, oa_total = 0.0, 0.0
    # class_total = torch.zeros(cfg['num_classes']).cpu()
    # class_count = torch.zeros(cfg['num_classes']).cpu()
    all_pred = np.empty(0).cpu()
    pb = trange(len(dataloader))
    # this is so we dont calc gradient bc not needed for val
    with torch.no_grad():
        for idx, (data, label, snr, extras) in enumerate(dataloader):
            data, label, snr = data.to(device), label.to(device), snr.to(device)
            if cfg['extra_params']:
                extras = extras.to(device)
            else:
                extras = None
            prediction = model(data, extras)
            loss = criterion(prediction, label, snr)
            
            loss_total += loss.item()
            
            pred_label = torch.argmax(prediction, dim=1)
            all_pred = np.append(all_pred, pred_label.cpu().numpy())
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
    
    return(loss_total, oa_total, all_pred)
    
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

        loss_train, oa_train, pred_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val, pred_val = validate(cfg, dl_val, model)
        
        scheduler.step()
        cr_train = met.classification_report(dl_train.dataset.species, pred_train, output_dict=True)
        cr_val = met.classification_report(dl_val.dataset.species, pred_val, output_dict=True)
        t_rep = {y: 
                 {x: cr_train[x][y] for x in ['0', '1', '2', '3', '4']}
                 for y in ['precision','recall']}
        v_rep = {y: 
                 {x: cr_val[x][y] for x in ['0', '1', '2', '3', '4']}
                 for y in ['precision','recall']}  
        writer.add_scalars('Train/prec', t_rep['precision'], current_epoch)
        writer.add_scalars('Val/prec', v_rep['precision'], current_epoch)
        writer.add_scalars('Train/recall', t_rep['recall'], current_epoch)
        writer.add_scalars('Val/recall', v_rep['recall'], current_epoch)
        writer.add_scalar('Train/AvgPrec', cr_train['macro avg']['precision'], current_epoch)
        writer.add_scalar('Val/AvgPrec', cr_val['macro avg']['precision'], current_epoch)
        writer.add_scalar('Train/OA', oa_train, current_epoch)
        writer.add_scalar('Train/Loss', loss_train, current_epoch)
        # writer.add_scalar('CBA/Train', np.mean(cba_train), current_epoch)
        writer.add_scalar('Val/OA', oa_val, current_epoch)
        writer.add_scalar('Val/Loss', loss_val, current_epoch)
        # writer.add_scalar('CBA/Val', np.mean(cba_val), current_epoch)
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
