from utils import AverageMeter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy


def client_update(client_model, optimizer, train_loader, epoch, num_batch, device='cuda'):
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

#%%

def client_train_model(model, data_loader, criterion, optimizer):
    """训练模型"""
    model.train()
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(data_loader)

def server_aggregate(global_model, client_models, data_size_weights):
    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        global_dict[k] = torch.stack([data_size_weights[i] * client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).sum(0) #+ sum([data_size_weights[idx] for idx in non_sampled_client_idx]) * global_model.state_dict()[k].float()

    global_model.load_state_dict(global_dict)


#%%

def average_models(models, weights):
    """对多个模型进行加权平均"""
    avg_model = copy.deepcopy(models[0])
    for param in avg_model.parameters():
        param.data.zero_()
    num_models = len(models)
    for i, model in enumerate(models):
        for avg_param, model_param in zip(avg_model.parameters(), model.parameters()):
            avg_param.data.add_(weights[i] * model_param.data)
    return avg_model

#%%

def test(global_model, test_loader, device='cuda'):
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