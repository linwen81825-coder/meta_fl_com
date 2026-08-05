import torch
import torch.nn as nn
from dataset.dataSplit_LN_new import get_data_loaders_new
from model.model import MLP
from model.wideresnet import SmallMetaConvNet, WideResNet, SmallMetaConvNet1 ,ResNet18
import datetime
from dataset.dataSplit_clothing1m import get_data_loaders_clothing1m
import argparse
import math


# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def build_model(dataset, layers=10, widen_factor=2, droprate=0):
def build_model(dataset):
#     if dataset == 'cifar10':
#         model = ResNet18(num_classes=10)

#     elif dataset == 'cifar100':
#         model = ResNet18(num_classes=100)

#     elif dataset == 'clothing1m':
#         model = SmallMetaConvNet1(num_classes=14)

#     else:
#         raise ValueError(f"Unsupported dataset: {dataset}")

#     if torch.cuda.is_available():
#         model.cuda()
#         torch.backends.cudnn.benchmark = True

#     return model

    # model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    # model = WideResNet(
    #     layers,
    #     dataset == 'cifar10' and 10 or 100,
    #     widen_factor,
    #     dropRate=droprate
    # )

    if dataset == 'cifar10':
        model = SmallMetaConvNet(num_classes=10)
    elif dataset == 'cifar100':
        model = SmallMetaConvNet(num_classes=100)
    elif dataset == 'clothing1m':
        model = SmallMetaConvNet1(num_classes=14)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


    # if dataset == 'cifar10':
    #     model = WideResNet(
    #         layers,
    #         10,
    #         widen_factor,
    #         dropRate=droprate
    #     )
    # elif dataset == 'cifar100':
    #     model = WideResNet(
    #         layers,
    #         100,
    #         widen_factor,
    #         dropRate=droprate
    #     )
    # elif dataset == 'clothing1m':
    #     model = SmallMetaConvNet1(num_classes=14)

    # if torch.cuda.is_available():
    #     model.cuda()
    #     torch.backends.cudnn.benchmark = True

    # return model


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

            test_loss += loss.item() * target.size(0)

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

    # Top-1 专家激活频率。
    # last_selected_expert shape: [batch_size]
    selected_expert = model.fc.last_selected_expert

    if selected_expert is None:
        raise RuntimeError(
            'model.fc.last_selected_expert is None.'
        )

    expert_activation_frequency = (
        torch.bincount(
            selected_expert.reshape(-1),
            minlength=model.fc.num_experts
        ).to(
            device=device,
            dtype=torch.float32
        )
        / data.size(0)
    )

    loss = criterion(output, target)
    model_params = tuple(model.params())
    # 计算权重更新量
    pseudo_grads = torch.autograd.grad(
        loss,
        model_params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True
    )

    pseudo_grads = tuple(
        torch.zeros_like(param)
        if grad is None
        else grad.detach()
        for param, grad in zip(
            model_params,
            pseudo_grads
        )
    )

    return (
        loss.detach(),
        expert_activation_frequency.detach(),
        pseudo_grads
    )

parser = argparse.ArgumentParser(
    description='Your script description.'
)

parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    help='The name of the dataset.'
)

parser.add_argument(
    '--use_dirichlet',
    type=str,
    default='true',
    help='Whether to use Dirichlet distribution for data splitting.'
)

parser.add_argument(
    '--dirichlet_alpha',
    type=float,
    default=0.1,
    help='Alpha parameter for the Dirichlet distribution.'
)

parser.add_argument(
    '--num_selected',
    type=int,
    default=20,
    help='Number of selected items.'
)

# 解析命令行参数
args = parser.parse_args()

use_dirichlet = args.use_dirichlet.lower() == 'true'
dirichlet_alpha = args.dirichlet_alpha
num_selected = args.num_selected
dataset = args.dataset
print('expert aggregation = mlp')
print('nonexpert aggregation = fedavg')
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
num_epochs = 1
num_rounds = 1000
num_batches = 1

# meta dataset parameters
meta_bs = 128
meta_sample_number = 1000

# FL model parameters
lr = 0.03
min_lr = 0.0003
decay_factor = 0.996

# Meta model parameters
meta_net_hidden_size = 500
meta_net_num_layers = 1
meta_lr = 1e-4
meta_weight_decay = 0

nesterov = True
momentum = 0.9
weight_decay = 5e-4


if dataset == 'clothing1m':
    (
        train_dataloaders,
        test_dataloader,
        meta_dataloader
    ) = get_data_loaders_clothing1m(
        num_clients,
        batch_size,
        meta_bs,
        meta_sample_number,
        use_dirichlet=use_dirichlet,
        dirichlet_alpha=dirichlet_alpha
    )
else:
    (
        train_dataloaders,
        test_dataloader,
        meta_dataloader
    ) = get_data_loaders_new(
        num_clients,
        batch_size,
        meta_bs,
        meta_sample_number,
        dataset=dataset,
        isnoise=True,
        use_dirichlet=use_dirichlet,
        dirichlet_alpha=dirichlet_alpha
    )


# 初始化模型和优化器
global_model = build_model(dataset)
num_experts = global_model.fc.num_experts
criterion = nn.CrossEntropyLoss()

optimizer_model = torch.optim.SGD(
    global_model.params(),
    lr,
    momentum=momentum,
    nesterov=nesterov,
    weight_decay=weight_decay
)


# 模拟多个客户端
client_model = build_model(dataset).to(device)

# 初始化元学习网络
meta_net = MLP(
    input_size=2,
    hidden_size=meta_net_hidden_size,
    num_layers=meta_net_num_layers,
    output_size=1
).to(device)

meta_optimizer = torch.optim.Adam(
    meta_net.parameters(),
    lr=meta_lr,
    weight_decay=meta_weight_decay
)

meta_dataloader_iter = iter(meta_dataloader)


# 获取当前时间
now = datetime.datetime.now()

# 格式化时间字符串
time_str = now.strftime('%m%d_%H%M')


# # 设置随机的通信成功率
# prob_vector = (
#     torch.rand(num_clients) * 0.7 + 0.3
# ).view(-1, 1).to(device)

# print("prob_vector: ", prob_vector)

best_acc = 0.0

fedavg_weight = torch.full(
    (num_clients,),
    1.0 / num_clients,
    dtype=torch.float32,
    device=device
)

for round in range(num_rounds):

    # 模型学习率余弦退火：从 lr 平滑衰减到 min_lr。
    current_lr = (
        min_lr
        + 0.5
        * (lr - min_lr)
        * (
            1.0
            + math.cos(
                math.pi * round / num_rounds
            )
        )
    )

    pseudo_net = build_model(dataset)

    pseudo_net.load_state_dict(
        global_model.state_dict()
    )

    # 客户端训练并上传权重更新
    client_losses = []
    client_expert_frequencies = []
    grads_list = []

    for i in range(num_clients):
        client_model.load_state_dict(
            pseudo_net.state_dict()
        )

        # 每个客户端使用新的独立优化器
        client_optimizer = torch.optim.SGD(
            client_model.params(),
            lr=current_lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )

        (
            loss,
            expert_activation_frequency,
            weight_updates
        ) = client_train_1(
            client_model,
            train_dataloaders[i],
            criterion,
            client_optimizer,
            num_epochs,
            num_batches
        )

        grads_list.append(weight_updates)
        client_losses.append(loss.item())
        client_expert_frequencies.append(
            expert_activation_frequency
        )

        del client_optimizer


    client_losses_tensor = torch.tensor(
        client_losses
    ).view(-1, 1).to(device)

    client_expert_frequencies_tensor = torch.stack(
        client_expert_frequencies,
        dim=0
    ).to(device)
    # [num_clients, num_experts]

    # 不对 loss 做标准化。
    # 对客户端 k、专家 e 构造：
    # [client_loss_k, activation_frequency_k_e]
    loss_features = (
        client_losses_tensor
        .unsqueeze(1)
        .expand(
            -1,
            num_experts,
            -1
        )
    )
    # [num_clients, num_experts, 1]

    frequency_features = (
        client_expert_frequencies_tensor
        .unsqueeze(-1)
    )
    # [num_clients, num_experts, 1]

    client_expert_features = torch.cat(
        [
            loss_features,
            frequency_features
        ],
        dim=2
    )
    # [num_clients, num_experts, 2]

    # 每个客户端-专家对输出一个客户端聚合权重分数。
    raw_expert_weights = meta_net(
        client_expert_features.reshape(
            -1,
            2
        )
    ).view(
        num_clients,
        num_experts
    )

    # 每个专家分别沿客户端维度归一化。
    # expert_weights[:, expert_id].sum() == 1
    expert_weights = (
        raw_expert_weights
        / raw_expert_weights.sum(
            dim=0,
            keepdim=True
        ).clamp_min(1e-12)
    )

    print(
        "raw_expert_weights:",
        raw_expert_weights
    )

    print(
        "expert_weights:",
        expert_weights
    )

    print(
        "expert_weight_sums:",
        expert_weights.sum(dim=0)
    )

    avg_client_loss = client_losses_tensor.mean().item()        



    # 聚合客户端梯度
    # ==================================================
    # 按专家参数和非专家参数分别选择聚合方式
    # ==================================================

    # 参数名称的顺序必须和 client_train_1() 中
    # model.params() 返回的梯度顺序一致
    param_names = [
        name
        for name, _ in pseudo_net.named_params(pseudo_net)
    ]

    if len(param_names) != len(grads_list[0]):
        raise RuntimeError(
            "参数名称数量与客户端梯度数量不一致"
        )

    aggregated_grads = []

    for param_index, param_name in enumerate(
        param_names
    ):
        # ----------------------------------------------
        # 1. 判断当前参数属于专家还是非专家
        # ----------------------------------------------
        if param_name.startswith("fc.experts."):
            # 参数名示例：
            # fc.experts.0.fc1.weight
            # fc.experts.1.fc2.bias
            expert_id = int(
                param_name.split('.')[2]
            )

            # 当前专家专属的客户端权重
            current_weights = (
                expert_weights[:, expert_id]
            )
        else:
            # 卷积层、BN参数、Gate和其他非专家参数
            # 继续使用客户端等权平均
            current_weights = fedavg_weight

        # ----------------------------------------------
        # 3. 聚合当前参数的所有客户端梯度
        # ----------------------------------------------
        aggregated_grad = torch.zeros_like(
            grads_list[0][param_index]
        )

        for client_id in range(num_clients):
            aggregated_grad += (
                grads_list[client_id][param_index]
                * current_weights[client_id]
            )

        aggregated_grads.append(
            aggregated_grad
        )


    # 更新伪模型
    pseudo_net.update_params(
        lr_inner=current_lr,
        source_params=aggregated_grads
    )


    # ==================================================
    # 更新元学习网络
    # ==================================================
    del aggregated_grads

    try:
        meta_inputs, meta_labels = next(
            meta_dataloader_iter
        )
    except StopIteration:
        meta_dataloader_iter = iter(
            meta_dataloader
        )

        meta_inputs, meta_labels = next(
            meta_dataloader_iter
        )

    meta_inputs = meta_inputs.to(device)
    meta_labels = meta_labels.to(device)

    meta_outputs = pseudo_net(meta_inputs)

    meta_loss = criterion(
        meta_outputs,
        meta_labels.long()
    )

    meta_optimizer.zero_grad()

    meta_loss.backward()

    meta_optimizer.step()

    print(
        'meta_loss: ',
        meta_loss
    )

    print(
        'meta_net.output_layer.weight after: ',
        meta_net.output_layer.weight[:, 0:5]
    )

    # 两种模式都更新全局模型
    global_model.load_state_dict(
        pseudo_net.state_dict()
    )


    test_train_losses = torch.arange(
        num_clients,
        dtype=torch.float32
    ).to(device)

    print(
        "test_train_losses: ",
        test_train_losses
    )

    test_train_losses = torch.reshape(
        test_train_losses,
        (len(test_train_losses), 1)
    )

    test_loss_features = (
        test_train_losses
        .unsqueeze(1)
        .expand(
            -1,
            num_experts,
            -1
        )
    )

    test_frequency_features = (
        client_expert_frequencies_tensor
        .unsqueeze(-1)
    )

    test_client_expert_features = torch.cat(
        [
            test_loss_features,
            test_frequency_features
        ],
        dim=2
    )

    with torch.no_grad():
        test_raw_expert_weights = meta_net(
            test_client_expert_features.reshape(
                -1,
                2
            )
        ).view(
            num_clients,
            num_experts
        )

        test_expert_weights = (
            test_raw_expert_weights
            / test_raw_expert_weights.sum(
                dim=0,
                keepdim=True
            ).clamp_min(1e-12)
        )

    print(
        'test_expert_weights:',
        test_expert_weights
    )

    print(
        'test_expert_weight_sums:',
        test_expert_weights.sum(dim=0)
    )

    test_loss, test_accuracy = test_model(
        global_model,
        test_dataloader,
        criterion
    )

    best_acc = max(best_acc, test_accuracy)

    print(
        f"Round {round + 1} | "
        f"Client Loss: {avg_client_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.2f}% | "
        f"Best Acc: {best_acc:.2f}%",
        flush=True
    )


    torch.save(
        meta_net.state_dict(),
        (
            f'./save/mlp_model_s_LN'
            f'{num_selected}_N{num_clients}_'
            f'BS{batch_size}_{dataset}_'
            f'{time_str}.pth'
        )
    )

    # 两种模式都保存全局模型
    torch.save(
        global_model.state_dict(),
        (
            f'./save/expert_mlp_nonexpert_fedavg_'
            f'global_model_s_LN'
            f'{num_selected}_N{num_clients}_'
            f'BS{batch_size}_{dataset}_'
            f'{time_str}.pth'
        )
    )