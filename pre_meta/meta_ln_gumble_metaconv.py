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
from util1 import *

# modules
from model import ToyCifarNet
from dataSplit_LN import get_data_loaders
from utils import AverageMeter
from labelNoiseEstimator import compute_ratio_per_client_update, get_auxiliary_data_loader
# from sampling import uniform_sampling
from scipy.stats import entropy

from duel_func import get_client_idx, update
from sampling import *
#from resnet import ResNet32, MetaLinear, MetaModule
#from resnet import ResNet32, MetaLinear, MetaModule# ,VNet
from load_corrupted_data import CIFAR10, CIFAR100

from sorting_operator import SortingOperator, SubsetOperator


from wideresnet import WideResNet # , VNet

def build_model(dataset, layers=10, widen_factor=10, droprate=0):
    # model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    model = WideResNet(layers, dataset == 'cifar10' and 10 or 100,
                       widen_factor, dropRate=droprate)
    # weights_init(model)

    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.params()])))

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

def cal_loss(data, model):
    
    with torch.no_grad():
        
        inputs_val, targets_val = data

        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)

        output = model(inputs_val)

        loss = F.cross_entropy(output, targets_val)
    
    return loss


def cal_loss1(data, model):
    
    with torch.no_grad():
        
        inputs_val, targets_val = data

        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)

        output = model(inputs_val)

        loss = F.cross_entropy(output, targets_val)
    
    return loss
def client_update(client_model, optimizer, train_loader, epoch, num_batch):
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
            
            
            
def server_aggregate(global_model, client_models, data_size_weights, num_selected):
    global_dict = global_model.state_dict()

    # non_sampled_client_idx = [i for i in list(range(num_clients)) if i not in list(client_idx)]

    for k in global_dict.keys():

        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float()/num_selected for i in range(num_selected)], 0).sum(0) #+ sum([data_size_weights[idx] for idx in non_sampled_client_idx]) * global_model.state_dict()[k].float()


    global_model.load_state_dict(global_dict)
'''
class VNet(MetaModule):
    def __init__(self, input, hidden1, output, num_selected):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        self.num_selected = num_selected
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        x = self.linear2(x)
        return F.sigmoid(x)
        #return torch.transpose(F.gumbel_softmax(torch.transpose(x, 0, 1), hard=False), 0, 1)
        #return torch.transpose(F.softmax(torch.transpose(x, 0, 1)), 0, 1)

 
class VNet(MetaModule):
    def __init__(self, input, hidden1, hidden2, output, num_selected):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, hidden2) 
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = MetaLinear(hidden2, output)
        
        self.num_selected = num_selected

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return F.sigmoid(x)    
 '''   


class VNet(MetaModule):
    def __init__(self, input, hidden1, output, num_selected):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        self.num_selected = num_selected
        self.k = num_selected
        self.subset_sample = SubsetOperator(k=num_selected, tau=1, hard=True)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        x = self.linear2(x)
        # return F.sigmoid(x)
        # print(x)
        return self.subset_sample(torch.transpose(x, 0, 1))
    
    
def loss_normalize(vec, amp):
    
    vec_max = torch.max(vec)
    vec_min = torch.min(vec)
    
    vec_norm = torch.zeros((len(vec), 1))
    
    for i in range(len(vec)):
        vec_norm[i] = (vec[i, 0] - vec_min) / (vec_max - vec_min) * amp
        
    return vec_norm

# GPU settings
torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

NUM_CLASSES = 10
# Parameters that can be tuned during different simulations
num_clients = 100
num_selected = 30 # init 20
num_rounds = 10000	# num of communication rounds

epochs = 1			# num of epochs in local client training (An epoch means you go through the entire dataset in that client)
batch_size = 10 # batch size already set in train dataloaderin dataSplit.py
num_batch = 1 # change here choose 100 - 300 default:50
real_epochs = 1
real_num_batch = 4

# hyperparameters of deep models
lr = 0.1 # learning rate
decay_factor = 0.996

losses_train = []
losses_test = []
acc_train = []
acc_test = []
client_idx_lst = []
T_pull_lst = []
T_pull = np.zeros(num_clients)

# ratio1 = 0.2 # 0.2,0.5,0.8
weighted1 = False # True or False
ds = 64 # 32,64,128,256

rand_part1 = False # equally split dataset
isnoise1 = True # add label noise
increase_factor = 0.1 # range of client sampling
order1 = False # loss order, False-increase, True-decrease

global_ratio = 1
M = 1

random = False

np.random.seed(220)
torch.manual_seed(220)
torch.cuda.manual_seed_all(220)


# load dataset
client_train_loader, test_loader, data_size_per_client = get_data_loaders(num_clients, batch_size, real_wd=True, weighted=weighted1, classes_pc=10, verbose=True, rand_part=rand_part1, isnoise=True)
data_size_weights = data_size_per_client / data_size_per_client.sum()

# aux_loader = get_auxiliary_data_loader(testset_extract = True, data_size = ds)

# get meta dataset 
t_l, train_meta_loader, test_l = build_dataset('cifar10', True, batch_size = 40)


nesterov = True
# build model, global model, client models and vnet model
dataset = 'cifar10'
momentum = 0.9
weight_decay = 5e-4
num_batchs = 1

global_model = build_model(dataset)

# client_meta_models = [build_model(dataset).to(device) for _ in range(num_clients)]

# vnet = VNet(1, 100, 1, num_selected).cuda()
vnet = VNet(1, num_clients, 1, num_selected).cuda()

if dataset == 'cifar10':
    num_classes = 10
if dataset == 'cifar100':
    num_classes = 100


#optimizer_model = torch.optim.SGD(global_model.params(), lr,
#                                  momentum=momentum, weight_decay=weight_decay)

optimizer_model = torch.optim.SGD(global_model.params(), lr,
                                  momentum=momentum, nesterov=nesterov,
                                  weight_decay=weight_decay)

#optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3,
#                             weight_decay=1e-4)

optimizer_vnet = torch.optim.SGD(vnet.params(), 1e-3,
                                  momentum=momentum, nesterov=nesterov,
                                  weight_decay=weight_decay)

train_meta_loader_iter = iter(train_meta_loader)

client_models = [build_model(dataset).to(device) for _ in range(num_selected)]


optimizer_client_model = [torch.optim.SGD(model.params(), lr,
                              momentum=momentum, nesterov=nesterov,
                              weight_decay=weight_decay) for model in client_models]


client_train_loader_iter=[]

for i in range(num_clients):
    client_train_loader_iter.append(iter(client_train_loader[i])) 
    

for r in range(num_rounds):
    
    print("----------------------")
    print("----------------------")
    print("r: ", r)
    # optimizer_vnet.param_groups[0]['lr'] = lr * (decay_factor ** r)
    for m in range(M):
        global_model.train()

        global_meta_model = build_model(dataset)
        global_meta_model.load_state_dict(global_model.state_dict())

        for model in client_models:
            model.load_state_dict(global_meta_model.state_dict())
        #print("global_meta_model 1:", global_meta_model.conv1.weight[0])
        #global_meta_model = nn.DataParallel(global_meta_model)

        client_train_loader_next=[]
        for i in range(num_clients):
            try:
                client_train_loader_next.append(next(client_train_loader_iter[i]))
            except StopIteration:
                client_train_loader_iter[i] = iter(client_train_loader[i])
                client_train_loader_next.append(next(client_train_loader_iter[i]))

        # cal loss of each client
        cost_set_v = torch.zeros(num_clients).to(device)
        # client excute

        for i in range(num_clients):  
            cost_set_v[i] = cal_loss(client_train_loader_next[i], global_meta_model)



        # receive training loss of each client
        # reshape loss
        cost_set_v = torch.reshape(cost_set_v, (len(cost_set_v), 1))

        cost_set_v = loss_normalize(cost_set_v, 50).to(device)
        # print("cost_set_v: ", cost_set_v)
        # get selected client index by vnet
        v_lambda = vnet(cost_set_v)
        v_lambda = torch.transpose(v_lambda, 0, 1)

        # print('cost_set_v: ', cost_set_v)
        # print('v_lambda: ', v_lambda)

        '''
        #print('global_meta_model.conv1.weight 1: ', global_meta_model.conv1.weight[0])
        # grad_set = []
        grads_sum = 0
        meta_lr = lr * ((0.1 ** int(r >= 80)) * (0.1 ** int(r >= 100)))   # For ResNet32

        for i in range(num_clients):
            #if v_lambda_hard[i].data == 1:
            meta_model.zero_grad()
            # loss = torch.sum(cost_set_v[i])
            if grads_sum == 0:
                grads_sum = torch.autograd.grad(cost_set_v[i], (global_meta_model.params()), create_graph=True)
            else:
                grads = torch.autograd.grad(cost_set_v[i], (global_meta_model.params()), create_graph=True)
                grads_sum = [a+b*v_lambda_hard[i] for a,b in zip(grads_sum, grads)]
                # client_meta_models[i].update_params(lr_inner=meta_lr, source_params=grads)
                # grad_set.append(grads)
        '''        
        # grad_set = []


        sampled_client_idx = [i for i in list(range(num_clients)) if v_lambda[i]==1]

        non_sampled_client_idx = [i for i in list(range(num_clients)) if v_lambda[i]==0]

        grads_sum = 0

        for i in range(num_selected):

            client_models[i].zero_grad()

            client_idx = sampled_client_idx[i]   

            inputs_val, targets_val = client_train_loader_next[client_idx]

            inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)

            output = client_models[i](inputs_val)

            loss = F.cross_entropy(output, targets_val)

            # print(i, client_idx, loss)

            if grads_sum == 0:

                grads_sum = torch.autograd.grad(loss, (client_models[i].params()), create_graph=True)
            else:

                grads = torch.autograd.grad(loss, (client_models[i].params()), create_graph=True)            
                grads_sum = [a+b*v_lambda[client_idx].data for a,b in zip(grads_sum, grads)]

        # print(grads_sum[0][0][0])
        z = torch.tensor([0.]).to(device)
        grads_zero = [a*z for a in grads_sum]

        for j in non_sampled_client_idx:

            grads_sum = [a+b*v_lambda[j] for a,b in zip(grads_sum, grads_zero)]

        grads_avg = [x/num_selected for x in grads_sum]
        meta_lr = lr * ((0.1 ** int(r >= 18000)) * (0.1 ** int(r >= 19000)))  # For WRN-28-10
        global_meta_model.update_params(lr_inner=meta_lr, source_params=grads_avg)

        #print('global_meta_model.conv1.weight 2: ', global_meta_model.conv1.weight[0])
        print('vnet.linear1.weight before: ', vnet.linear1.weight[0:5])

        del grads_avg
        # get val dataset batch
        try:
            data_val, target_val = next(train_meta_loader_iter)
        except StopIteration:        
            train_meta_loader_iter = iter(train_meta_loader)
            data_val, target_val = next(train_meta_loader_iter)

        data_val, target_val = data_val.to(device), target_val.to(device)


        # get loss of meta model on validation dataset
        y_g_hat = global_meta_model(data_val)

        l_g_meta = F.cross_entropy(y_g_hat, target_val)
        print("l_g_meta: ", l_g_meta)
        # update vnet by meta loss
        optimizer_vnet.zero_grad()
        # print('vnet grad: ', vnet.linear1.weight.grad)
        l_g_meta.backward()
        optimizer_vnet.step()

        print('vnet.linear1.weight after: ', vnet.linear1.weight[0:5])

        test_loss, acc = test(global_meta_model, test_loader)
        # print("test_loss: ", test_loss, "acc: ", acc)
    
    global_model.load_state_dict(global_meta_model.state_dict())
    
    '''
    with torch.no_grad():
        w_new = vnet(cost_set_v)
        
    w_new = torch.transpose(w_new, 0, 1)
    
    sampled_client_idx = [i for i in list(range(num_clients)) if w_new[i]==1]
            
    loss = 0
    
    for i in range(num_selected):
        
        client_idx = sampled_client_idx[i]
        
        loss += client_update(client_models[i], optimizer_client_model[i], client_train_loader[client_idx], real_epochs, real_num_batch)
        T_pull[i] += 1
    
    loss = loss / num_selected
    
    server_aggregate(global_model, client_models, data_size_weights, num_selected)
    
    # print('vnet.linear1.weight after: ', vnet.linear1.weight[0:5])

    # print('cost_w_set_v: ', cost_w_set_v)
    print('w_new: ', w_new)
    # print('w_new_hard: ', w_new_hard)
    print('training loss: ', loss)
    
    test_loss, acc = test(global_model, test_loader)
    print("test_loss: ", test_loss, "acc: ", acc)
    
    cost_w_set = torch.zeros(num_clients).to(device)

    for i in range(num_clients):
        cost_w_set[i] = cal_loss_client1(global_model, client_train_loader[i], num_batchs)

    # reshape loss
    cost_w_set_v = torch.reshape(cost_w_set, (len(cost_w_set), 1))


    with torch.no_grad():
        w_new = vnet(cost_w_set_v)
    
        
    w_new_hard = torch.clone(w_new)
    clip = torch.topk(torch.transpose(w_new, 0, 1), num_selected)[1][0]
    # print(x)
    for i in range(w_new.shape[0]):
        if i in clip:
            w_new_hard[i] = 1
        else:
            w_new_hard[i] = 0

    loss = torch.sum(cost_w_set_v*w_new_hard)/num_selected
    
    # print('vnet.linear1.weight before: ', vnet.linear1.weight[0:5])
    
    # update global model
    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()
    
    # print('vnet.linear1.weight after: ', vnet.linear1.weight[0:5])

    print('cost_w_set_v: ', cost_w_set_v)
    print('w_new: ', w_new)
    print('w_new_hard: ', w_new_hard)
    print('loss: ', loss)
    
    test_loss, acc = test(global_model, test_loader)
    print("test_loss: ", test_loss, "acc: ", acc)
    
    '''
    #T_pull_lst.append(np.copy(T_pull))
    losses_test.append(test_loss)
    acc_test.append(acc)
    
    name = "./mat/0816_metaLN_gumble_lr"+str(lr)+"_C"+str(num_clients)+"_S"+str(num_selected)+"_Nbatch"+str(real_num_batch)+"_M"+str(M)+".mat"
    sio.savemat(name, {'acc_test': acc_test, 'losses_test':losses_test, 'T_pull_lst':T_pull_lst})
    #'''
