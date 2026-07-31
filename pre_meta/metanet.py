# Python3 Code
# Three Water's Coding Style
# edited and tested by Akida-Sho Wong
#
# log
#
# @Oct 3 11:19:18am begins
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
from dataSplit_LN import get_data_loaders
from utils import AverageMeter
from labelNoiseEstimator import compute_ratio_per_client_update, get_auxiliary_data_loader
# from sampling import uniform_sampling
from scipy.stats import entropy

from duel_func import get_client_idx, update
from sampling import *
from resnet import ResNet32, MetaLinear, MetaModule# ,VNet
#from load_corrupted_data import CIFAR10, CIFAR100

class VNet(MetaModule):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)
        #return F.sigmoid(out)
        return F.gumbel_softmax(torch.transpose(out, 0, 1), hard=True)

    
# GPU settings
torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

NUM_CLASSES = 10
# Parameters that can be tuned during different simulations
num_clients = 50
num_selected = 1 # init 20
num_rounds = 1000	# num of communication rounds

epochs = 5			# num of epochs in local client training (An epoch means you go through the entire dataset in that client)
batch_size = 100 # batch size already set in train dataloaderin dataSplit.py
num_batch = 10 # change here choose 100 - 300 default:50

# hyperparameters of deep models
lr = 0.1 # learning rate
decay_factor = 0.996

losses_train = []
losses_test = []
acc_train = []
acc_test = []
client_idx_lst = []
T_pull_lst = []

# ratio1 = 0.2 # 0.2,0.5,0.8
weighted1 = False # True or False
ds = 64 # 32,64,128,256

rand_part1 = False # equally split dataset
isnoise1 = True # add label noise
increase_factor = 0.1 # range of client sampling
order1 = False # loss order, False-increase, True-decrease


# FedAvg
def main():
	# data loader
	## client_train_loader: a list with #clients of **local** data loaders
	## test_loader: test.py only on **global** testset (local doesn't own seperate test.py data)

    client_train_loader, test_loader, data_size_per_client = get_data_loaders(num_clients, batch_size, real_wd=True, weighted=weighted1, classes_pc=10, verbose=True, rand_part=rand_part1, isnoise=isnoise1)
    data_size_weights = data_size_per_client / data_size_per_client.sum()
    
    aux_loader = get_auxiliary_data_loader(testset_extract = True, data_size = ds)
    # model configurations
    # ASSUME that global model and client models are with exactly same structures
    # ASSUME that num_select is constant during FedAvg
    global_model = ToyCifarNet().to(device)
    client_models = [ToyCifarNet(init_weights=False).to(device) for _ in range(num_selected)]
    
    vnet = VNet(1, 100, 1).cuda()
    optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3, weight_decay=1e-4)
    
    ## client models initialized by global model
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


    opt_lst = [optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4) for model in client_models]
    
    # for each communication round
    for r in range(num_rounds):
        print('----------------------------------------------------------------------')
        print('----------------------------------------------------------------------')

        for opt in opt_lst:
            opt.param_groups[0]['lr'] = lr * (decay_factor ** r)

        # cal training loss of each client
        client_loss = eval_clients(num_clients, global_model, client_train_loader).to(device)
        print('client_loss: ', client_loss)
        # select device by metanet
        client_indi = vnet(client_loss)
        client_idx = (client_indi==1).nonzero()[:,1]
        client_idx = client_idx.cpu().numpy()

        loss = 0
        print(client_idx)
        # update selected model
        for i in range(num_selected):
            loss += client_update(client_models[i], opt_lst[i], client_train_loader[client_idx[i]], epochs)

        # loss needed to average across selected clients
        losses_train.append(loss / num_selected)

        # updated local models send back for server aggregate
        server_aggregate(global_model, client_models, data_size_weights, client_idx)

        # return loss and acc on testset
        test_loss, acc = test(global_model, test_loader)
        losses_test.append(test_loss)
        acc_test.append(acc)
        
        # update metanet by training loss
        update_val_model(global_model, vnet, optimizer_vnet, aux_loader)


        print('%d-th round' % r)
        print('average train loss %0.3g | test.py loss %0.3g | test.py acc: %0.3f' % (loss / num_selected, test_loss, acc))
        
        name = "./mat/0707_metanet_lr"+str(lr)+"_decay"+str(decay_factor)+"_C"+str(num_clients)+"_S"+str(num_selected)+"_Nbatch"+str(num_batch)+"_weighted"+str(weighted1)+"_DataSize"+str(ds)+".mat"

        sio.savemat(name, {'acc_test': acc_test})


        
def eval_clients(num_clients, global_model, client_train_loader):
    tensor = torch.ones(())
    client_loss = tensor.new_empty((num_clients,1))
    # get loss of each client by global model
    for i in range(num_clients):
        client_loss[i,0],_ = client_eval(global_model, client_train_loader[i])
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


def update_val_model(train_model, vnet, optimizer_vnet, val_dataloader):
    
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



if __name__ == '__main__':
	main()
