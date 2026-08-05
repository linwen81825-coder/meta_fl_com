import copy
import datetime
import os
import argparse
import math

import torch
import torch.nn as nn

from dataset.dataSplit_LN_new import get_data_loaders_new
from dataset.dataSplit_clothing1m import get_data_loaders_clothing1m
from model.wideresnet import (
    SmallMetaConvNet,
    ResNet18,
)


device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

def build_model(dataset, layers=10, widen_factor=1, droprate=0):
    if dataset == 'cifar10':
        model = SmallMetaConvNet(num_classes=10)
    elif dataset == 'cifar100':
        model = SmallMetaConvNet(num_classes=100)
    elif dataset == 'clothing1m':
        model = SmallMetaConvNet1(num_classes=14)
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model
# def build_model(
#     dataset,
#     layers=10,
#     widen_factor=1,
#     droprate=0
# ):
#     if dataset == 'cifar10':
#         model = ResNet18(
#             num_classes=10,
#             num_experts=4,
#             expert_hidden_dim=128
#         )

#     elif dataset == 'cifar100':
#         model = ResNet18(
#             num_classes=100,
#             num_experts=4,
#             expert_hidden_dim=128
#         )

#     elif dataset == 'clothing1m':
#         model = SmallMetaConvNet1(
#             num_classes=14
#         )

#     else:
#         raise ValueError(
#             f'Unsupported dataset: {dataset}'
#         )

#     model = model.to(device)

#     if torch.cuda.is_available():
#         torch.backends.cudnn.benchmark = True

#     return model


def optimizer_state_to_cpu(obj):
    """
    将 optimizer.state_dict() 中的张量递归搬到 CPU。

    这样只在 GPU 上保留当前客户端的 optimizer state，
    其他客户端的 momentum buffer 保存在 CPU。
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()

    if isinstance(obj, dict):
        return {
            key: optimizer_state_to_cpu(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            optimizer_state_to_cpu(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            optimizer_state_to_cpu(value)
            for value in obj
        )

    return copy.deepcopy(obj)


def client_train(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs,
    num_batches,
    global_state_cpu
):
    """
    单个客户端每轮训练 num_batches 个 batch。

    本地模型进入该客户端前已经加载 global_model。
    训练结束后，返回：
      1. 客户端平均训练 loss；
      2. local_state - global_state 的完整更新量。

    更新量直接搬到 CPU，避免把所有客户端更新长期留在显存中。
    """
    model.train()

    train_loss = 0.0
    trained_batches = 0

    for _ in range(num_epochs):
        for batch_idx, (data, target) in enumerate(
            train_loader
        ):
            if batch_idx >= num_batches:
                break

            data = data.to(
                device,
                non_blocking=True
            )
            target = target.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            trained_batches += 1

    if trained_batches == 0:
        raise RuntimeError(
            'Client train loader has no batch.'
        )

    weight_updates = {}

    for name, value in model.state_dict().items():
        current_cpu = value.detach().cpu()
        reference_cpu = global_state_cpu[name]

        weight_updates[name] = (
            current_cpu - reference_cpu
        )

    return (
        train_loss / trained_batches,
        weight_updates
    )


def accumulate_weight_updates(
    update_sums,
    weight_updates
):
    """
    在线累加客户端更新，不保存 100 份完整 updates_list。

    浮点张量按原 dtype 累加；
    整型 buffer 使用 float32 累加，最终平均后再 round。
    """
    for name, update in weight_updates.items():
        if update.is_floating_point() or update.is_complex():
            if name not in update_sums:
                update_sums[name] = torch.zeros_like(
                    update
                )

            update_sums[name].add_(update)

        else:
            if name not in update_sums:
                update_sums[name] = torch.zeros_like(
                    update,
                    dtype=torch.float32
                )

            update_sums[name].add_(
                update.float()
            )


def average_weight_updates(
    update_sums,
    reference_state,
    num_clients
):
    """
    将更新总和转换成 FedAvg 等权平均更新。
    """
    aggregated_updates = {}

    for name, update_sum in update_sums.items():
        reference = reference_state[name]

        if (
            reference.is_floating_point()
            or reference.is_complex()
        ):
            aggregated_update = (
                update_sum / num_clients
            ).to(reference.dtype)

        else:
            aggregated_update = (
                update_sum / num_clients
            ).round().to(reference.dtype)

        aggregated_updates[name] = (
            aggregated_update
        )

    return aggregated_updates


def update_model(
    model,
    aggregated_updates
):
    """
    将 CPU 上的平均更新写回全局模型。
    """
    model_state = model.state_dict()

    with torch.no_grad():
        for name, value in model_state.items():
            update = aggregated_updates[name].to(
                device=value.device,
                dtype=value.dtype
            )

            value.add_(update)


def test_model(
    model,
    test_loader,
    criterion
):
    model.eval()

    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(
                device,
                non_blocking=True
            )
            target = target.to(
                device,
                non_blocking=True
            )

            output = model(data)
            loss = criterion(output, target)

            test_loss += (
                loss.item() * target.size(0)
            )

            pred = output.argmax(
                dim=1,
                keepdim=True
            )

            correct += pred.eq(
                target.view_as(pred)
            ).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = (
        100.0
        * correct
        / len(test_loader.dataset)
    )

    return test_loss, accuracy


parser = argparse.ArgumentParser(
    description=(
        'FedAvg with one reusable local model.'
    )
)

parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10'
)

parser.add_argument(
    '--use_dirichlet',
    type=str,
    default='true'
)

parser.add_argument(
    '--dirichlet_alpha',
    type=float,
    default=0.1
)

parser.add_argument(
    '--num_selected',
    type=int,
    default=20,
    help=(
        '保留命令行兼容性；'
        '当前 FedAvg 每轮使用全部客户端。'
    )
)

args = parser.parse_args()

use_dirichlet = (
    args.use_dirichlet.lower() == 'true'
)

dirichlet_alpha = args.dirichlet_alpha
dataset = args.dataset

print('dataset =', dataset)

if dataset == 'clothing1m':
    num_clients = 20
    batch_size = 32
else:
    num_clients = 100
    batch_size = 64

num_epochs = 1
num_rounds = 1000
num_batches = 1

meta_bs = 128
meta_sample_number = 1000

lr = 0.03
min_lr = 0.0003
nesterov = True
momentum = 0.9
weight_decay = 5e-4

if dataset == 'clothing1m':
    (
        train_dataloaders,
        test_dataloader,
        _
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
        _
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


global_model = build_model(dataset)
local_model = build_model(dataset)

criterion = nn.CrossEntropyLoss()

# 每个客户端的 SGD momentum 状态保存在 CPU。
# GPU 同一时刻只存在一个客户端 optimizer。
client_optimizer_states = [
    None
    for _ in range(num_clients)
]

os.makedirs(
    './save',
    exist_ok=True
)

now = datetime.datetime.now()
time_str = now.strftime('%m%d_%H%M')

best_accuracy = 0.0

for round_idx in range(num_rounds):
    # 余弦退火：从原始 lr 平滑衰减到 min_lr。
    current_lr = (
        min_lr
        + 0.5
        * (lr - min_lr)
        * (
            1.0
            + math.cos(
                math.pi * round_idx / num_rounds
            )
        )
    )

    # 本轮所有客户端都以同一份全局模型为起点。
    global_state_gpu = global_model.state_dict()

    # 用于在 CPU 上计算每个客户端的完整更新量。
    global_state_cpu = {
        name: value.detach().cpu().clone()
        for name, value in global_state_gpu.items()
    }

    client_losses = []
    update_sums = {}

    for client_id in range(num_clients):
        # 复用同一个本地模型。
        local_model.load_state_dict(
            global_state_gpu,
            strict=True
        )

        # 每次只创建当前客户端的 optimizer。
        local_optimizer = torch.optim.SGD(
            local_model.params(),
            lr=current_lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )

        # 恢复该客户端上一轮的 momentum，
        # 保持与“每客户端一个 optimizer”的原代码语义一致。
        if (
            client_optimizer_states[client_id]
            is not None
        ):
            local_optimizer.load_state_dict(
                client_optimizer_states[client_id]
            )

        # load_state_dict 会恢复上一轮保存的学习率，
        # 因此覆盖为当前轮余弦退火学习率。
        for param_group in local_optimizer.param_groups:
            param_group['lr'] = current_lr

        (
            train_loss,
            weight_updates
        ) = client_train(
            local_model,
            train_dataloaders[client_id],
            criterion,
            local_optimizer,
            num_epochs,
            num_batches,
            global_state_cpu
        )

        client_losses.append(train_loss)

        # 在线累加，不保存完整 updates_list。
        accumulate_weight_updates(
            update_sums,
            weight_updates
        )

        # 将该客户端 optimizer 状态保存回 CPU。
        client_optimizer_states[client_id] = (
            optimizer_state_to_cpu(
                local_optimizer.state_dict()
            )
        )

        del local_optimizer
        del weight_updates

    average_train_loss = (
        sum(client_losses)
        / len(client_losses)
    )

    aggregated_updates = (
        average_weight_updates(
            update_sums,
            global_state_cpu,
            num_clients
        )
    )

    update_model(
        global_model,
        aggregated_updates
    )

    test_loss, test_accuracy = test_model(
        global_model,
        test_dataloader,
        criterion
    )

    best_accuracy = max(
        best_accuracy,
        test_accuracy
    )

    print(
        f'Round {round_idx + 1} | '
        f'Client Loss: {average_train_loss:.4f} | '
        f'Test Loss: {test_loss:.4f} | '
        f'Test Acc: {test_accuracy:.2f}% | '
        f'Best Acc: {best_accuracy:.2f}%',
        flush=True
    )

    torch.save(
        global_model.state_dict(),
        (
            './save/'
            'global_model_s_LN_fedavg_'
            'resnet18_moe_single_local_'
            f'N{num_clients}_'
            f'BS{batch_size}_'
            f'{dataset}_'
            f'{time_str}.pth'
        )
    )
