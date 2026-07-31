# edit at 10.7.2021
import os
from tqdm import tqdm

import random
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset

# modules
from model import ToyCifarNet
#from dataSplit_LN import get_data_loaders
from utils import AverageMeter
from labelNoiseEstimator import compute_ratio_per_client_update, get_auxiliary_data_loader
# from sampling import uniform_sampling
from scipy.stats import entropy

from duel_func import get_client_idx, update
from sampling import *
from resnet import ResNet32, MetaLinear, MetaModule
#from resnet import ResNet32, MetaLinear, MetaModule# ,VNet
from load_corrupted_data import CIFAR10, CIFAR100


def build_dataset(dataset, augment, num_meta=1000, corruption_prob=0, corruption_type='unif', seed=1, batch_size=100, prefetch=0):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if dataset == 'cifar10':
        train_data_meta = CIFAR10(
            root='../../data', train=True, meta=True, num_meta=num_meta, corruption_prob=corruption_prob,
            corruption_type=corruption_type, transform=train_transform, download=True)
        train_data = CIFAR10(
            root='../../data', train=True, meta=False, num_meta=num_meta, corruption_prob=corruption_prob,
            corruption_type=corruption_type, transform=train_transform, download=True, seed=seed)
        test_data = CIFAR10(root='../../data', train=False, transform=test_transform, download=True)


    elif dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root='../../data', train=True, meta=True, num_meta=num_meta, corruption_prob=corruption_prob,
            corruption_type=corruption_type, transform=train_transform, download=True)
        train_data = CIFAR100(
            root='../../data', train=True, meta=False, num_meta=num_meta, corruption_prob=corruption_prob,
            corruption_type=corruption_type, transform=train_transform, download=True, seed=seed)
        test_data = CIFAR100(root='../../data', train=False, transform=test_transform, download=True)

    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=batch_size, shuffle=True,
        num_workers=prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                              num_workers=prefetch, pin_memory=True)

    return train_loader, train_meta_loader, test_loader




def eval_clients(num_clients, global_model, client_train_loader):
    tensor = torch.ones(())
    client_loss = tensor.new_empty((num_clients,1))
    # get loss of each client by global model
    for i in range(num_clients):
        client_loss[i,0],_ = client_eval(global_model, client_train_loader[i])
        # print('client ',i, 'client_loss: ', client_loss)
    # print(client_loss)
    return client_loss        
        

    
def client_eval(model, data_loader):
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()

    model.eval()

    with torch.no_grad():

        for data, target in data_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = F.cross_entropy(output, target)
            loss_avg.update(loss.item(), data.size(0))


            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            acc_avg.update(pred.eq(target.view_as(pred)).sum().item(), data.size(0))

    return loss_avg.avg, acc_avg.avg


def update_val_model(train_model, val_model, optimizer_vnet, val_dataloader):
    
    for data, target in val_dataloader:

        data, target = data.to(device), target.to(device)

        output = train_model(data)

        l_g_meta = F.cross_entropy(output, target)
        
        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()
        
        

# actually standard training in a local client/device
def client_update(client_model, optimizer, train_loader, epoch):
    loss_avg = AverageMeter()

    client_model.train()

    for e in range(epoch):
        # batch size from data loader is set to 32
        for batch_idx, (data, target) in enumerate(train_loader):

            if batch_idx == num_batch:
                break

            # transfer a mini-batch to GPU
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = client_model(data)
            # here use F.nll_loss() is wrong
            loss = F.cross_entropy(output, target)

            loss_avg.update(loss.item(), data.size(0))

            loss.backward()

            optimizer.step()



    return loss_avg.avg # average loss in this client over entire trainset over multiple epochs

def server_aggregate(global_model, client_models, data_size_weights, client_idx):
    global_dict = global_model.state_dict()

    non_sampled_client_idx = [i for i in list(range(num_clients)) if i not in list(client_idx)]

    for k in global_dict.keys():

        global_dict[k] = torch.stack([data_size_weights[client_idx[i]] * client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).sum(0) + sum([data_size_weights[idx] for idx in non_sampled_client_idx]) * global_model.state_dict()[k].float()


    global_model.load_state_dict(global_dict)



    # step of 'send aggregated global model to client' is here
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def test(global_model, test_loader):
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    global_model.eval()

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = global_model(data)

            loss = F.cross_entropy(output, target)
            loss_avg.update(loss.item(), data.size(0))


            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            acc_avg.update(pred.eq(target.view_as(pred)).sum().item(), data.size(0))

    return loss_avg.avg, acc_avg.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def build_model(dataset):
    model = ResNet32(dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


        
        
        
def cal_loss_client(model, data_loader):
    
    
    
    for batch_idx, (data, target) in enumerate(data_loader):
        
        if batch_idx == 1:
            break
            
        data, target = data.to(device), target.to(device)

        output = model(data)
        
        loss = F.cross_entropy(output, target)
        
        
    return loss    
