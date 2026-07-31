import torch
import torch.nn as nn
from dataset.dataSplit_LN_new import get_data_loaders_new
from model.wideresnet import SmallMetaConvNet, WideResNet, SmallMetaConvNet1
import datetime
from dataset.dataSplit_clothing1m import get_data_loaders_clothing1m
import argparse
import numpy as np
from itertools import chain, combinations


class LearningWithFairnessGuarantee:
    def __init__(self, N, m, w, r, eta):
        """
        初始化LFG算法的参数
        :param N: 手臂的数量
        :param m: 最大同时选择的手臂数量
        :param w: 每个手臂的权重列表
        :param r: 每个手臂的最小选择比例列表
        :param eta: 算法中的学习率参数
        """
        self.N = N
        self.m = m
        self.w = w
        self.r = r
        self.eta = eta

        # 初始化变量
        self.h = np.zeros(N)  # h_i(t): 手臂i被选择的次数
        self.Q = np.zeros(N)  # Q_i(t): 虚拟队列长度
        self.d = np.zeros(N)  # d_i(t): 手臂i在t轮被选择的指示符
        self.X = np.zeros(N)  # X_i(t): 手臂i在t轮的奖励
        self.mu_hat = np.zeros(N)  # μ̂_i(t): 到t轮为止观测到的手臂i的奖励样本均值
        self.mu_bar = np.ones(N)  # μ̄_i(t): t轮中手臂i的UCB估计

    def update_ucb(self, i, t):
        """
        更新手臂i的UCB估计 μ̄_i(t)
        :param i: 手臂的索引
        :param t: 当前轮次
        """
        if self.h[i] > 0:
            self.mu_bar[i] = min(self.mu_hat[i] + np.sqrt(3 * np.log(t) / (2 * self.h[i])), 1)
        else:
            self.mu_bar[i] = 1

    def update_virtual_queue(self, i):
        """
        更新手臂i的虚拟队列长度 Q_i(t)
        :param i: 手臂的索引
        """
        self.Q[i] = max(self.Q[i] + self.r[i] - self.d[i], 0)

    def select_super_arm(self, A_t):
        """
        选择超手臂S(t)
        :param A_t: 当前可用的手臂集合
        :return: 选择的超手臂S(t)
        """
        # 计算每个手臂的选择值
        arm_values = [(i, self.Q[i] + self.eta * self.w[i] * self.mu_bar[i]) for i in A_t]

        # 按照计算的选择值降序排序
        sorted_arms = sorted(arm_values, key=lambda x: x[1], reverse=True)

        # 选择前m个手臂
        selected_arm = [arm[0] for arm in sorted_arms[:self.m]]

        return selected_arm

    def update_estimates(self, S_t, rewards):
        """
        更新手臂的选择次数和奖励样本均值
        :param S_t: 选择的超手臂S(t)
        :param rewards: 超手臂S(t)对应的奖励
        """
        for i in range(self.N):
            if i in S_t:
                self.d[i] = 1
                self.h[i] += 1
                self.X[i] = rewards[S_t.index(i)]
                self.mu_hat[i] = (self.mu_hat[i] * (self.h[i] - 1) + self.X[i]) / self.h[i]
            else:
                self.d[i] = 0


# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(dataset, layers=10, widen_factor=1, droprate=0):
    # model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    # model = WideResNet(layers, dataset == 'cifar10' and 10 or 100, widen_factor, dropRate=droprate)
    if dataset == 'cifar10':
        model = SmallMetaConvNet(num_classes=10)
    elif dataset == 'cifar100':
        model = SmallMetaConvNet(num_classes=100)
    elif dataset == 'clothing1m':
        model = SmallMetaConvNet1(num_classes=14)
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


parser = argparse.ArgumentParser(description='Your script description.')
parser.add_argument('--dataset', type=str, default='clothing1m', help='The name of the dataset.')
parser.add_argument('--use_dirichlet', type=str, default='false', help='Whether to use Dirichlet distribution for data splitting.')
parser.add_argument('--dirichlet_alpha', type=float, default=0.5, help='Alpha parameter for the Dirichlet distribution.')
parser.add_argument('--num_selected', type=int, default=20, help='Number of selected items.')

# 解析命令行参数
args = parser.parse_args()
use_dirichlet = args.use_dirichlet.lower() == 'true'  # 转换为布尔值
dirichlet_alpha = args.dirichlet_alpha
num_selected = args.num_selected
dataset = args.dataset

# 使用参数


print('dataset = ', dataset)
if dataset == 'clothing1m':
    num_clients = 20
    num_selected = 5
    batch_size = 32
else:
    num_clients = 100
    num_selected = num_selected
    batch_size = 64
# 联邦学习训练和测试
num_epochs = 1  # 客户端本地训练的epoch数
num_rounds = 10000  # 联邦学习的轮数
num_batches = 1

# meta dataset parameters
meta_bs = 128
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

if dataset == 'clothing1m':
    train_dataloaders, test_dataloader, meta_dataloader = get_data_loaders_clothing1m(num_clients, batch_size, meta_bs,
                                                                                      meta_sample_number,
                                                                                      use_dirichlet=use_dirichlet,
                                                                                      dirichlet_alpha=0.5)
else:
    train_dataloaders, test_dataloader, meta_dataloader = get_data_loaders_new(num_clients, batch_size, meta_bs,
                                                                               meta_sample_number,
                                                                               dataset=dataset,
                                                                               isnoise=True,
                                                                               use_dirichlet=use_dirichlet,
                                                                               dirichlet_alpha=0.5)


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
# 设置随机的通信成功率
prob_vector = (torch.rand(num_clients) * 0.01 + 0.99).view(-1, 1).to(device)
print("prob_vector: ", prob_vector)

# 初始化 MAB 算法
# N = num_clients  # 手臂数量等于客户端数量
# T = num_rounds  # 时间地平线等于联邦学习的轮数
# m = num_selected  # 每轮选择的客户端数量
w = np.ones(num_clients)  # 假设每个客户端的权重相同
r = np.ones(num_clients) / (num_clients)  # 假设每个客户端的最小选择比例相同
eta = 0.1  # 设置学习率参数
lfg = LearningWithFairnessGuarantee(num_clients, num_selected, w, r, eta)

A_t = list(range(num_clients))  # 当前可用的手臂集合（客户端）
for round in range(num_rounds):

    for model in client_models:
        model.load_state_dict(global_model.state_dict())
    # 客户端训练并上传权重更新
    for i in range(num_clients):
        loss = client_train(client_models[i], train_dataloaders[i], criterion, client_optimizers[i],
                                              num_epochs, num_batches)

    # 使用 MAB 算法选择客户端

    for i in A_t:
        lfg.update_ucb(i, round)
        lfg.update_virtual_queue(i)
    # 使用 LFG 算法选择客户端集合
    selected_clients = lfg.select_super_arm(A_t)
    selected_clients_list = list(selected_clients)

    # 生成 multi-hot 向量
    pseudo_weight = torch.zeros(num_clients, device=device)
    pseudo_weight[selected_clients] = 1

    # 模拟客户端上传成功与否
    isTransmitted = torch.bernoulli(prob_vector).cpu().numpy()
    rewards = isTransmitted[selected_clients_list]  # 真实奖励为所选客户端是否成功上传

    # 更新 LFG 算法的奖励
    lfg.update_estimates(selected_clients_list, rewards)

    # 服务器聚合更新
    true_update = pseudo_weight.data * (torch.tensor(isTransmitted, device=device).transpose(0, 1).float())
    print("True Update Num: ", true_update)
    server_aggregate1(global_model, client_models, pseudo_weight)
    print('estimate reward: ', lfg.mu_hat)

    test_loss, test_accuracy = test_model(global_model, test_dataloader, criterion)
    print(f"Round {round + 1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # torch.save(meta_net.state_dict(), f'/home/xm/code/meta_model/random_model_s_COM_LN_hard_S{num_selected}_N{num_clients}_BS{batch_size}_{time_str}.pth')
    # torch.save(global_model.state_dict(), f'/home/xm/code/meta_model/random_global_model_SMC_COM_LN_hard_S{num_selected}_N{num_clients}_BS{batch_size}_{dataset}_{time_str}.pth')

