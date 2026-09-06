import copy
import datetime
import os
import argparse
import math
import sys
import atexit
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.resnet18_half_moe import ResNet18HalfMoE
from model.wideresnet_moe import WideResNetMoE
from model.shallow_resnet5_moe import ShallowResNet5MoE
from model.shallow_resnet3_moe import ShallowResNet3MoE
from model.small_meta_convnet_gn import SmallMetaConvNetGN
from model.small_meta_convnet_dropout import SmallMetaConvNetDropout
from model.small_meta_convnet_layernorm import SmallMetaConvNetLayerNorm
from model.small_meta_convnet_residual import SmallMetaConvNetResidual
from model.resnet_moe import ResNet10MoE, ResNet18MoE
from dataset.dataSplit_LN_new import get_data_loaders_new
from dataset.dataSplit_clothing1m import get_data_loaders_clothing1m
from model.wideresnet import (
    SmallMetaConvNet,
    ResNet18,
    ResNet20Lite,
    MetaResNetConvNet,
    MetaMobileNetV2Lite,
    MetaDenseNetLite,
    SmallMetaMobileNetV2,
    SmallMetaDenseNet,
    SmallMetaEfficientNet,
    SmallMetaVGG
)


device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)


class RoundOnlyConsoleLogger:
    """
    所有 stdout 内容都写入日志文件；
    只有以 "Round " 开头的完整行同时显示在控制台。
    """

    def __init__(self, log_path, console_stream):
        self.log_path = log_path
        self.console_stream = console_stream
        self.log_stream = open(
            log_path,
            mode='a',
            encoding='utf-8',
            buffering=1
        )
        self.pending_text = ''
        self._closed = False

    @property
    def closed(self):
        return self._closed

    @property
    def encoding(self):
        return getattr(
            self.console_stream,
            'encoding',
            'utf-8'
        )

    def isatty(self):
        return False

    def fileno(self):
        return self.console_stream.fileno()

    def write(self, text):
        if self._closed:
            return 0

        if not isinstance(text, str):
            text = str(text)

        # 完整内容始终写入日志。
        if not self.log_stream.closed:
            self.log_stream.write(text)

        # 控制台只输出每轮汇总行。
        self.pending_text += text

        while '\n' in self.pending_text:
            line, self.pending_text = (
                self.pending_text.split('\n', 1)
            )

            if line.startswith('Round '):
                self.console_stream.write(
                    line + '\n'
                )
                self.console_stream.flush()

        return len(text)

    def flush(self):
        # 解释器退出时可能在 close() 后再次调用 flush()。
        if not self.log_stream.closed:
            self.log_stream.flush()

        try:
            self.console_stream.flush()
        except (ValueError, OSError):
            pass

    def close(self):
        # 允许 atexit 和解释器清理阶段重复调用。
        if self._closed:
            return

        self.flush()

        if not self.log_stream.closed:
            self.log_stream.close()

        self._closed = True




def build_model(dataset):
    if dataset in ['cifar10','cinic10']:
        # model = SmallMetaConvNet(num_classes=10)
        model = SmallMetaVGG(num_classes=10,num_experts=4,expert_hidden_dim=128)
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
    global_state_cpu,
    load_balance_coef
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

            cross_entropy_loss = criterion(
                output,
                target
            )

            load_balance_loss = (
                model.fc.last_load_balance_loss
            )

            if load_balance_loss is None:
                raise RuntimeError(
                    'model.fc.last_load_balance_loss is None.'
                )

            training_loss = (
                cross_entropy_loss
                + load_balance_coef
                * load_balance_loss
            )

            training_loss.backward()

            # 只对专家参数按“实际路由到该专家的样本数”做内部平均。
            selected_experts = (
                model.fc.last_selected_experts
            )

            if selected_experts is None:
                raise RuntimeError(
                    'model.fc.last_selected_experts is None.'
                )

            expert_routed_counts = torch.bincount(
                selected_experts.reshape(-1),
                minlength=model.fc.num_experts
            ).to(
                device=device,
                dtype=torch.float32
            )

            batch_sample_count = float(
                data.size(0)
            )

            for param_name, param in (
                model.named_parameters()
            ):
                if not param_name.startswith(
                    'fc.experts.'
                ):
                    continue

                if param.grad is None:
                    continue

                expert_id = int(
                    param_name.split('.')[2]
                )

                routed_count = (
                    expert_routed_counts[expert_id]
                )

                if routed_count.item() > 0:
                    param.grad.mul_(
                        batch_sample_count
                        / routed_count
                    )
                else:
                    param.grad.zero_()

            optimizer.step()

            # Client Loss 仍然记录纯交叉熵。
            train_loss += cross_entropy_loss.item()
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
    '--imbalanced_factor',
    type=float,
    default=2.581,
    help='Long-tail imbalance factor. None disables long-tail sampling.'
)

parser.add_argument(
    '--num_selected',
    type=int,
    default=100,
    help=(
        '保留命令行兼容性；'
        '当前 FedAvg 每轮使用全部客户端。'
    )
)

parser.add_argument(
    '--cache_client_train_loaders',
    type=str,
    default='false',
    choices=['true', 'false'],
    help=(
        'Whether to reuse each client DataLoader '
        'across global rounds. '
        'true preserves the original behavior.'
    )
)

parser.add_argument(
    '--loader_seed',
    type=int,
    default=1,
    help=(
        'DataLoader random seed corresponding to '
        'bayes/meta data.seed. It does not change '
        'the existing data partition/noise seed.'
    )
)

parser.add_argument(
    '--deterministic',
    type=str,
    default='true',
    choices=['true', 'false'],
    help='Use deterministic per-loader generator/worker seeding.'
)

parser.add_argument(
    '--num_workers',
    type=int,
    default=0,
    help='Number of DataLoader workers.'
)

parser.add_argument(
    '--pin_memory',
    type=str,
    default='false',
    choices=['true', 'false'],
    help='Enable DataLoader pin_memory on CUDA.'
)

parser.add_argument(
    '--prefetch_factor',
    type=int,
    default=2,
    help='Per-worker prefetch factor; used only when num_workers > 0.'
)

args = parser.parse_args()

use_dirichlet = (
    args.use_dirichlet.lower() == 'true'
)

dirichlet_alpha = args.dirichlet_alpha
imbalanced_factor = args.imbalanced_factor
dataset = args.dataset

cache_client_train_loaders = (
    args.cache_client_train_loaders.lower()
    == 'true'
)

loader_seed = int(args.loader_seed)
deterministic = (
    args.deterministic.lower() == 'true'
)
num_workers = int(args.num_workers)
pin_memory = (
    args.pin_memory.lower() == 'true'
)
prefetch_factor = int(
    args.prefetch_factor
)

# 自动保存完整日志；控制台只显示每轮 Round 汇总行。
log_start_time = datetime.datetime.now()
log_time_str = log_start_time.strftime(
    '%m%d_%H%M%S'
)

os.makedirs(
    './log_long_noniid',
    exist_ok=True
)

log_file_path = (
    f'./log_long_noniid/FedAvg_LN_s_cache_resnet_'
    f'{dataset}_'
    f'{log_time_str}.log'
)

original_stdout = sys.stdout
round_console_logger = RoundOnlyConsoleLogger(
    log_file_path,
    original_stdout
)

sys.stdout = round_console_logger
atexit.register(
    round_console_logger.close
)

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

lr = 0.025
min_lr = 0.001
nesterov = False
momentum = 0.000001
weight_decay = 5e-4

# Top-2 MoE 负载均衡辅助损失系数。
load_balance_coef = 0.0

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
        isnoise=False,
        use_dirichlet=use_dirichlet,
        dirichlet_alpha=dirichlet_alpha,
        imbalanced_factor=imbalanced_factor
    )


# ==================================================
# Client train DataLoader cache (bayes/meta semantics)
# ==================================================
# get_data_loaders_*() 仍然只执行一次，因此原有客户端数据划分、
# label noise 和 Dataset 均保持不变。
#
# 这里仅重建客户端 train DataLoader，并严格复现 bayes/meta 的
# loader 随机性语义：
#
#   bayes client_id = 1, 2, ..., N
#   seed_offset = 1000 + client_id
#   loader seed = loader_seed + seed_offset
#
# cache_client_train_loaders=True:
#   每个客户端 DataLoader 只创建一次。
#   因为同一个 torch.Generator 被持续复用，所以 shuffle RNG
#   状态会跨 global round 连续推进。
#
# cache_client_train_loaders=False:
#   每次客户端训练前重新创建 DataLoader。
#   generator 会重新以该客户端固定 seed 初始化，因此每轮
#   都从相同的客户端 shuffle RNG 初始状态开始。
client_train_datasets = [
    loader.dataset
    for loader in train_dataloaders
]

# 原 get_data_loaders_*() 创建的客户端 loader 不再参与后续训练；
# 只保留其中已经固定的数据集对象。
train_dataloaders = None


def seed_worker(worker_id):
    # 与 bayes/meta 一致：DataLoader worker seed 由 generator 派生，
    # 并同步到 numpy / Python random。
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_client_train_loader(
    client_id
):
    # 当前脚本 client_id 是 0-based；
    # bayes/meta 的 client_id 是 1-based。
    bayes_client_id = (
        int(client_id) + 1
    )

    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': (
            pin_memory
            and device.type == 'cuda'
        ),
    }

    if deterministic:
        generator = torch.Generator()

        generator.manual_seed(
            loader_seed
            + 1000
            + bayes_client_id
        )

        loader_kwargs[
            'worker_init_fn'
        ] = seed_worker

        loader_kwargs[
            'generator'
        ] = generator

    if num_workers > 0:
        # 与 bayes/meta 一致：worker 模式下长期复用 worker，
        # 并启用配置的预取因子。
        loader_kwargs[
            'persistent_workers'
        ] = True

        loader_kwargs[
            'prefetch_factor'
        ] = prefetch_factor

    return DataLoader(
        client_train_datasets[
            client_id
        ],
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs
    )


client_train_loader_cache = {}

if cache_client_train_loaders:
    client_train_loader_cache = {
        client_id: (
            build_client_train_loader(
                client_id
            )
        )
        for client_id in range(
            num_clients
        )
    }


def get_client_train_loader(
    client_id
):
    if cache_client_train_loaders:
        return (
            client_train_loader_cache[
                client_id
            ]
        )

    return build_client_train_loader(
        client_id
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

# 训练开始时只记录一次实验配置。
num_experts = getattr(
    global_model.fc,
    'num_experts',
    'N/A'
)

experiment_config = [
    "EXPERIMENT_CONFIG_BEGIN",
    "method=fedavg",
    f"start_time={now.strftime('%Y-%m-%d_%H:%M:%S')}",
    f"log_file={log_file_path}",
    f"device={device}",
    f"dataset={dataset}",
    f"use_dirichlet={use_dirichlet}",
    f"dirichlet_alpha={dirichlet_alpha}",
    f"imbalanced_factor={imbalanced_factor}",
    f"num_clients={num_clients}",
    f"clients_per_round={num_clients}",
    f"num_selected_arg={args.num_selected}",
    f"batch_size={batch_size}",
    f"num_epochs={num_epochs}",
    f"num_batches={num_batches}",
    f"num_rounds={num_rounds}",
    f"cache_client_train_loaders={cache_client_train_loaders}",
    f"num_cached_client_loaders={len(client_train_loader_cache)}",
    f"loader_seed={loader_seed}",
    f"deterministic_loader={deterministic}",
    f"num_workers={num_workers}",
    f"pin_memory={pin_memory}",
    f"prefetch_factor={prefetch_factor}",
    "client_loader_seed_rule=loader_seed+1000+(client_id+1)",
    f"meta_batch_size={meta_bs}",
    f"meta_sample_number={meta_sample_number}",
    f"model={global_model.__class__.__name__}",
    f"num_experts={num_experts}",
    "routing=top2",
    "top_k=2",
    "top2_output=renormalized_gate_weighted_sum",
    "aggregation=equal_average_full_client_updates",
    "client_training_loss=cross_entropy+load_balance",
    "client_loss_log=cross_entropy_only",
    f"lr_max={lr}",
    f"lr_min={min_lr}",
    "lr_scheduler=cosine_annealing",
    f"optimizer=SGD",
    f"momentum={momentum}",
    f"nesterov={nesterov}",
    f"weight_decay={weight_decay}",
    f"load_balance_coef={load_balance_coef}",
    "client_optimizer_state=persistent_per_client_on_cpu",
    "EXPERIMENT_CONFIG_END",
]

print(
    "\n".join(experiment_config),
    flush=True
)

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

        train_loader = get_client_train_loader(
            client_id
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
            train_loader,
            criterion,
            local_optimizer,
            num_epochs,
            num_batches,
            global_state_cpu,
            load_balance_coef
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

    # torch.save(
    #     global_model.state_dict(),
    #     (
    #         './save/'
    #         'global_model_s_LN_fedavg_'
    #         'resnet18_moe_single_local_'
    #         f'N{num_clients}_'
    #         f'BS{batch_size}_'
    #         f'{dataset}_'
    #         f'{time_str}.pth'
    #     )
    # )
