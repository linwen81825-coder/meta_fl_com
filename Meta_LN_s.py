import torch
import torch.nn as nn
from dataset.dataSplit_LN_new import get_data_loaders_new
from model.model import MLP
from model.wideresnet import SmallMetaConvNet, WideResNet, SmallMetaConvNet1
import datetime
from dataset.dataSplit_clothing1m import get_data_loaders_clothing1m
import argparse


# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(dataset, layers=10, widen_factor=1, droprate=0):
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


parser = argparse.ArgumentParser(
    description='Your script description.'
)

parser.add_argument(
    '--dataset',
    type=str,
    default='clothing1m',
    help='The name of the dataset.'
)

parser.add_argument(
    '--use_dirichlet',
    type=str,
    default='false',
    help='Whether to use Dirichlet distribution for data splitting.'
)

parser.add_argument(
    '--dirichlet_alpha',
    type=float,
    default=0.5,
    help='Alpha parameter for the Dirichlet distribution.'
)

parser.add_argument(
    '--num_selected',
    type=int,
    default=20,
    help='Number of selected items.'
)

# 新增：选择 MLP 加权或 FedAvg 等权平均
parser.add_argument(
    '--aggregation',
    type=str,
    default='mlp',
    choices=['mlp', 'fedavg'],
    help='Aggregation method: mlp or fedavg.'
)


# 解析命令行参数
args = parser.parse_args()

use_dirichlet = args.use_dirichlet.lower() == 'true'
dirichlet_alpha = args.dirichlet_alpha
num_selected = args.num_selected
dataset = args.dataset
aggregation = args.aggregation


print('dataset = ', dataset)
print('aggregation = ', aggregation)


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
lr = 0.1
decay_factor = 0.996

# Meta model parameters
meta_net_hidden_size = 500
meta_net_num_layers = 1
meta_lr = 1e-4
meta_weight_decay = 0.

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

criterion = nn.CrossEntropyLoss()

optimizer_model = torch.optim.SGD(
    global_model.params(),
    lr,
    momentum=momentum,
    nesterov=nesterov,
    weight_decay=weight_decay
)


# 模拟多个客户端
client_models = [
    build_model(dataset).to(device)
    for _ in range(num_clients)
]


client_optimizers = [
    torch.optim.SGD(
        model.params(),
        lr,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay
    )
    for model in client_models
]


# 初始化元学习网络
meta_net = MLP(
    hidden_size=meta_net_hidden_size,
    num_layers=meta_net_num_layers
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


# 设置随机的通信成功率
prob_vector = (
    torch.rand(num_clients) * 0.7 + 0.3
).view(-1, 1).to(device)

print("prob_vector: ", prob_vector)


for round in range(num_rounds):

    pseudo_net = build_model(dataset)

    pseudo_net.load_state_dict(
        global_model.state_dict()
    )

    for model in client_models:
        model.load_state_dict(
            pseudo_net.state_dict()
        )

    # 客户端训练并上传权重更新
    client_losses = []
    grads_list = []

    for i in range(num_clients):
        loss, weight_updates = client_train_1(
            client_models[i],
            train_dataloaders[i],
            criterion,
            client_optimizers[i],
            num_epochs,
            num_batches
        )

        grads_list.append(weight_updates)
        client_losses.append(loss)


    client_losses_tensor = torch.tensor(
        client_losses
    ).view(-1, 1).to(device)

    print(
        'Average_train_loss: ',
        client_losses_tensor.mean()
    )


    # ==================================================
    # 根据参数选择聚合权重
    # ==================================================
    if aggregation == 'mlp':

        # MLP 输出形状为 [num_clients, 1]
        raw_weights = meta_net(
            client_losses_tensor.data
        ).squeeze(1)

        # 归一化为权重和等于 1
        pseudo_weight = (
            raw_weights
            / raw_weights.sum().clamp_min(1e-12)
        )

        print("raw_weights:", raw_weights)
        print("pseudo_weight:", pseudo_weight)
        print("weight_sum:", pseudo_weight.sum())

    elif aggregation == 'fedavg':

        # FedAvg：所有客户端等权平均
        pseudo_weight = torch.full(
            (num_clients,),
            1.0 / num_clients,
            device=device
        )

        print(
            "FedAvg weight:",
            pseudo_weight[0]
        )

        print(
            "weight_sum:",
            pseudo_weight.sum()
        )


    # 聚合客户端梯度
    aggregated_grads = [
        torch.zeros_like(grad)
        for grad in grads_list[0]
    ]

    for grads, weight in zip(
        grads_list,
        pseudo_weight
    ):
        for i, grad in enumerate(grads):
            aggregated_grads[i] += (
                grad * weight
            )


    # 更新伪模型
    pseudo_net.update_params(
        lr_inner=lr,
        source_params=aggregated_grads
    )


    # ==================================================
    # 只有 MLP 模式更新元学习网络
    # ==================================================
    if aggregation == 'mlp':

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
            'meta_net.linear1.weight after: ',
            meta_net.output_layer.weight[0, 0:5]
        )

    else:
        del aggregated_grads


    # 两种模式都更新全局模型
    global_model.load_state_dict(
        pseudo_net.state_dict()
    )


    # 只有 MLP 模式打印元网络输出
    if aggregation == 'mlp':

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

        print(
            'test_out:',
            meta_net(test_train_losses)
        )


    test_loss, test_accuracy = test_model(
        global_model,
        test_dataloader,
        criterion
    )

    print(
        f"Round {round + 1}, "
        f"Test Loss: {test_loss:.4f}, "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )


    # MLP 模式保存元网络
    if aggregation == 'mlp':
        torch.save(
            meta_net.state_dict(),
            (
                f'/save/mlp_model_s_LN'
                f'{num_selected}_N{num_clients}_'
                f'BS{batch_size}_{dataset}_'
                f'{time_str}.pth'
            )
        )

    # 两种模式都保存全局模型
    torch.save(
        global_model.state_dict(),
        (
            f'/save/{aggregation}_'
            f'global_model_s_LN'
            f'{num_selected}_N{num_clients}_'
            f'BS{batch_size}_{dataset}_'
            f'{time_str}.pth'
        )
    )