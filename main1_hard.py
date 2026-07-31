import torch
import torch.nn as nn
from dataset.dataSplit_LN_new import get_data_loaders_new
from model import MLPH
from model.wideresnet import WideResNet

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

def client_train(model, train_loader, criterion, optimizer, num_epochs, num_batches):
    model.train()
    initial_params = {name: param.clone() for name, param in model.state_dict().items()}
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / num_batches)

    # 计算权重更新量
    weight_updates = {name: param - initial_params[name] for name, param in model.state_dict().items()}
    return weight_updates, sum(train_losses) / num_epochs


# 定义服务器聚合函数
def aggregate_weight_updates(updates_list):
    # 初始化聚合后的权重更新
    aggregated_updates = {name: torch.zeros_like(updates_list[0][name]) for name in updates_list[0].keys()}

    for updates in updates_list:
        for name, update in updates.items():
            aggregated_updates[name] += update

    # 计算平均权重更新
    for name in aggregated_updates.keys():
        aggregated_updates[name] /= len(updates_list)

    return aggregated_updates


# 定义全局模型更新函数
def update_model(model, aggregated_updates):
    with torch.no_grad():
        for name, param in model.state_dict().items():
            param += aggregated_updates[name]


# 定义模型测试函数
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


def client_train_1(model, train_loader, criterion, optimizer, num_epochs, num_batches):
    model.train()
    # 只训练一个epoch和一个batch
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)

    # 计算权重更新量
    pseudo_grads = torch.autograd.grad(loss, model.params(), create_graph=True)
    return loss, pseudo_grads


num_clients = 10
num_selected = 5
# 联邦学习训练和测试
num_epochs = 1  # 客户端本地训练的epoch数
num_rounds = 10000  # 联邦学习的轮数
num_batches = 1
batch_size = 32
# meta dataset parameters
meta_bs = 32
meta_sample_number = 1000

# FL model parameters
lr = 0.1  # learning rate
decay_factor = 0.996

# Meta model parameters
meta_net_hidden_size = 100
meta_net_num_layers = 1
meta_lr = 1e-4
meta_weight_decay = 0.

nesterov = True
momentum = 0.9
weight_decay = 5e-4

dataset = 'cifar10'

train_dataloaders, test_dataloader, meta_dataloader = get_data_loaders_new(num_clients, batch_size, meta_bs, meta_sample_number)

# 初始化模型和优化器
global_model = build_model(dataset)
criterion = nn.CrossEntropyLoss()
optimizer_model = torch.optim.SGD(global_model.params(), lr,
                                  momentum=momentum, nesterov=nesterov,
                                  weight_decay=weight_decay)
# 模拟多个客户端
client_models = [build_model(dataset).to(device) for _ in range(num_clients)]


client_optimizers = [torch.optim.SGD(model.params(), lr,
                              momentum=momentum, nesterov=nesterov,
                              weight_decay=weight_decay) for model in client_models]

# 初始化元学习网络
meta_net = MLPH(hidden_size=meta_net_hidden_size, num_layers=meta_net_num_layers, k=num_selected, hard=True).to(device)
meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=meta_lr, weight_decay=meta_weight_decay)

meta_dataloader_iter = iter(meta_dataloader)

for round in range(num_rounds):

    pseudo_net = build_model(dataset)
    pseudo_net.load_state_dict(global_model.state_dict())

    for model in client_models:
        model.load_state_dict(pseudo_net.state_dict())
    # 客户端训练并上传权重更新
    client_losses = []
    grads_list = []
    for i in range(num_clients):
        loss, weight_updates = client_train_1(client_models[i], train_dataloaders[i], criterion, client_optimizers[i],
                                              num_epochs, num_batches)
        grads_list.append(weight_updates)
        client_losses.append(loss)



    client_losses_tensor = torch.tensor(client_losses).view(-1, 1).to(device)
    print("client_losses: ", client_losses_tensor.data)
    pseudo_weight = meta_net(client_losses_tensor.data)
    print("pseudo_weight: ", pseudo_weight)

    # 聚合权重更新并更新全局模型
    # 初始化聚合后的权重更新
    aggregated_grads = [torch.zeros_like(grads_list[0][i]) for i in range(len(grads_list[0]))]
    # 加权平均
    for grads, weight in zip(grads_list, pseudo_weight[0]):
        for i in range(len(grads)):
            aggregated_grads[i] += grads[i] * weight

    # 计算平均权重更新
    total_weight = pseudo_weight.sum()
    for i in range(len(aggregated_grads)):
        aggregated_grads[i] /= total_weight
    # 更新全局模型
    pseudo_net.update_params(lr_inner=lr, source_params=aggregated_grads)

    # 更新元学习模型
    del aggregated_grads
    # get val dataset batch
    try:
        meta_inputs, meta_labels = next(meta_dataloader_iter)
    except StopIteration:
        meta_dataloader_iter = iter(meta_dataloader)
        meta_inputs, meta_labels = next(meta_dataloader_iter)

    meta_inputs, meta_labels = meta_inputs.to(device), meta_labels.to(device)
    meta_outputs = pseudo_net(meta_inputs)
    meta_loss = criterion(meta_outputs, meta_labels.long())

    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()

    print('meta_net.linear1.weight after: ', meta_net.output_layer.weight[0, 0:5])

    global_model.load_state_dict(pseudo_net.state_dict())
    if round == 500:
        print(round)
    x = torch.arange(num_clients, dtype=torch.float32).to(device)
    x = torch.reshape(x, (len(x), 1))
    print('test_out:', meta_net(x))

    test_loss, test_accuracy = test_model(global_model, test_dataloader, criterion)
    print(f"Round {round + 1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

