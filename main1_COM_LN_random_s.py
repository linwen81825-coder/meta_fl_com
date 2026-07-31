import torch
import torch.nn as nn
from dataset.dataSplit_LN_new import get_data_loaders_new
from model.wideresnet import SmallMetaConvNet, WideResNet
import datetime

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(dataset, layers=10, widen_factor=1, droprate=0):
    # model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    model = WideResNet(layers, dataset == 'cifar10' and 10 or 100, widen_factor, dropRate=droprate)
    # model = SmallMetaConvNet()
    # weights_init(model)

    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.params()])))

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


def client_train(model, train_loader, criterion, optimizer, num_epochs, num_batches):
    model.train()
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

    return sum(train_losses) / num_epochs


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


def server_aggregate1(global_model, client_models, client_weights):
    # Initialize an empty dictionary to hold the aggregated model parameters
    global_dict = global_model.state_dict()

    # Loop over each parameter in the global model
    for k in global_dict.keys():
        # Initialize the parameter aggregation with zeros
        global_dict[k] = torch.zeros_like(global_dict[k])

        # Sum up the weighted parameters from each client model
        for i, client_model in enumerate(client_models):
            client_state_dict = client_model.state_dict()
            global_dict[k] += client_weights[i] * client_state_dict[k].float()

    # Normalize the aggregated parameters by the sum of weights to ensure correct averaging
    total_weight = sum(client_weights)
    for k in global_dict.keys():
        global_dict[k] /= total_weight

    # Load the aggregated parameters back into the global model
    global_model.load_state_dict(global_dict)

    # Update each client model with the aggregated global model
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

    return global_model


# def client_train(model, train_loader, criterion, optimizer, num_epochs, num_batches):
#     model.train()  # Set the model to training mode
#
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         # Iterate over batches in the train_loader
#         for batch_idx, (data, target) in enumerate(train_loader):
#             if batch_idx >= num_batches:
#                 break  # Stop after processing the specified number of batches
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             running_loss += loss.item()
#
#         # Calculate the average loss for this epoch
#         epoch_loss = running_loss / num_batches
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
#
#     # Return the final loss and model parameters for further processing
#     return epoch_loss
def generate_khot_vector(n, k, device='cpu'):
    """
    生成一个k-hot向量，长度为n，其中k个位置被随机设置为1，其余为0。

    参数:
    n (int): 向量的长度。
    k (int): 需要被设置为1的位置数。
    device (str): 指定向量存放的设备，默认为'cpu'。

    返回:
    torch.Tensor: 生成的k-hot向量。
    """
    # 初始化一个全零的向量
    vector = torch.zeros(n, device=device)

    # 确保k不大于n
    if k > n:
        raise ValueError("k must be less than or equal to n")

        # 如果没有位置需要设置为1，则直接返回全零向量
    if k == 0:
        return vector

        # 生成k个不重复的索引，这些索引会被设置为1
    indices = torch.randperm(n, device=device)[:k]

    # 在这些索引位置上将值设置为1
    vector[indices] = 1

    return vector



num_clients = 100
num_selected = 20
# 联邦学习训练和测试
num_epochs = 1  # 客户端本地训练的epoch数
num_rounds = 5000  # 联邦学习的轮数
num_batches = 1
batch_size = 64
# meta dataset parameters
meta_bs = 128
meta_sample_number = 1000

# FL model parameters
lr = 0.1  # learning rate
decay_factor = 0.996

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


# 获取当前时间
now = datetime.datetime.now()
# 格式化时间字符串
time_str = now.strftime('%m%d_%H%M')

alpha = 1 / num_clients

for round in range(num_rounds):

    # 设置随机的通信成功率
    prob_vector = (torch.rand(num_clients) * 0.7 + 0.3).view(-1, 1).to(device)
    print("prob_vector: ", prob_vector)

    for model in client_models:
        model.load_state_dict(global_model.state_dict())
    # 客户端训练并上传权重更新
    for i in range(num_clients):
        loss = client_train(client_models[i], train_dataloaders[i], criterion, client_optimizers[i],
                                              num_epochs, num_batches)

    pseudo_weight = generate_khot_vector(num_clients, num_selected, device=device)

    isTransmitted = torch.bernoulli(prob_vector).transpose(0, 1)
    client_com_prob_r_tensor_clip = torch.clip(isTransmitted, min=1e-1)
    print("isTransmitted: ", isTransmitted)
    print("pseudo_weight: ", pseudo_weight)
    true_update = pseudo_weight.data * isTransmitted.data
    print("True Update Num: ", true_update)

    server_aggregate1(global_model, client_models, true_update[0])

    test_loss, test_accuracy = test_model(global_model, test_dataloader, criterion)
    print(f"Round {round + 1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # torch.save(meta_net.state_dict(), f'/home/xm/code/meta_model/random_model_s_COM_LN_hard_S{num_selected}_N{num_clients}_BS{batch_size}_{time_str}.pth')
    torch.save(global_model.state_dict(), f'/home/xm/code/meta_model/random_global_model_WRN_COM_LN_hard_S{num_selected}_N{num_clients}_BS{batch_size}_{time_str}.pth')

