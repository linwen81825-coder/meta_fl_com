import torch
import torch.nn as nn
from dataset.dataSplit_LN_new import get_data_loaders_new
from model.model import MLPH
from model.wideresnet import WideResNet, ResNet18
from utils.data_util import get_datetime

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(dataset, layers=10, widen_factor=1, droprate=0):
    # model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    model = WideResNet(10)
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
num_selected = 6
# 联邦学习训练和测试
num_epochs = 1  # 客户端本地训练的epoch数
num_rounds = 5000  # 联邦学习的轮数
num_batches = 1
batch_size = 32
# meta dataset parameters
meta_bs = 64
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

train_dataloaders, test_dataloader, meta_dataloader = get_data_loaders_new(num_clients, batch_size, meta_bs, meta_sample_number, isnoise=True)

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
# 通信成功率
prob_vector = torch.rand(num_clients).view(-1, 1).to(device)
# prob_vector = prob_vector / 2 + 0.5
print("com_prob: ", prob_vector)
# print("client_losses: ", client_losses_tensor.data)
print("prob_vector: ", prob_vector)

time_str = get_datetime()

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

    #client_com_prob_tensor = torch.tensor(prob_vector).view(-1, 1).to(device)
    # 本次传输成功与否，此处设置为失败的概率

    # features = torch.tensor(list(zip(client_losses_tensor, prob_vector)), requires_grad=True).float().to(device)

    pseudo_weight = meta_net(client_losses_tensor.data)
    # isTransmitted = torch.bernoulli(prob_vector).transpose(0, 1)
    # client_com_prob_r_tensor_clip = torch.clip(isTransmitted, min=1e-5)
    # print("isTransmitted: ", isTransmitted)
    print("pseudo_weight: ", pseudo_weight)
    # true_update = pseudo_weight.data * isTransmitted.data
    # print("True Update Num: ", true_update)

    # 聚合权重更新并更新全局模型
    # 初始化聚合后的权重更新
    aggregated_grads = [torch.zeros_like(grads_list[0][i]) for i in range(len(grads_list[0]))]
    # 加权平均
    success_client = 0
    for grads, weight in zip(grads_list, pseudo_weight[0]):#, client_com_prob_r_tensor_clip[0]):
        # if weight.data * isSuccess.data == 1:
        #     success_client += 1
        # print(isSuccess)
        for i in range(len(grads)):
            aggregated_grads[i] += grads[i] * weight #* isSuccess

    # success_rate = success_client / num_selected
    # print("success_rate: ", success_rate)
    #
    # if success_rate == 0:
    #     continue

    # 计算平均权重更新
    #total_weight = client_com_prob_r_tensor.sum()
    for i in range(len(aggregated_grads)):
        aggregated_grads[i] /= num_selected
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

    #if round <= 50:
    global_model.load_state_dict(pseudo_net.state_dict())
    # if round > 50:
    #     print(round)

    test_comm_probs = torch.arange(0.1, 0.1 * num_clients + 0.1, 0.1).to(device)
    # test_comm_probs = torch.rand(num_clients).to(device)
    test_train_losses = torch.arange(num_clients).to(device)  # 随机生成通信成功概率
    print("test_comm_probs: ", test_comm_probs)
    print("test_train_losses: ", test_train_losses)

    # test_features = torch.stack([test_train_losses, test_comm_probs], dim=1)
    # with torch.no_grad():
    #     test_out = meta_net(test_features)
    #     print('test_out:', test_out)
    x = torch.arange(num_clients, dtype=torch.float32).to(device)
    x = torch.reshape(x, (len(x), 1))
    print('test_out:', meta_net(x))

    test_loss, test_accuracy = test_model(global_model, test_dataloader, criterion)
    print(f"Round {round + 1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # torch.save(meta_net.state_dict(), f'/home/xm/code/meta_model/mlph_model_LN_hard_S{num_selected}_N{num_clients}_BS{batch_size}_{time_str}.pth')
    # torch.save(global_model.state_dict(), f'/home/xm/code/meta_model/global_model_LN_hard_S{num_selected}_N{num_clients}_BS{batch_size}_{time_str}.pth')

