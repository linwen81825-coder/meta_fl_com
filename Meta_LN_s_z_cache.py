import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataSplit_LN_new import get_data_loaders_new
from model.model import MLP
from model.wideresnet import (
    SmallMetaConvNet,
    WideResNet,
    SmallMetaConvNet1,
    ResNet18,
)
import datetime
from dataset.dataSplit_clothing1m import get_data_loaders_clothing1m
import argparse
import math
import os
import sys
import atexit
import random

import numpy as np


# 检查是否有可用的GPU
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

        self.log_stream.write(text)
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
        if not self.log_stream.closed:
            self.log_stream.flush()

        try:
            self.console_stream.flush()
        except (ValueError, OSError):
            pass

    def close(self):
        if self._closed:
            return

        self.flush()

        if not self.log_stream.closed:
            self.log_stream.close()

        self._closed = True


def build_model(dataset):
    if dataset == 'cifar10':
        model = SmallMetaConvNet(
            num_classes=10
        )
    elif dataset == 'cifar100':
        model = SmallMetaConvNet(
            num_classes=100
        )
    elif dataset == 'clothing1m':
        model = SmallMetaConvNet1(
            num_classes=14
        )
    else:
        raise ValueError(
            f'Unsupported dataset: {dataset}'
        )

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


def client_train(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs,
    num_batches
):
    model.train()

    initial_params = {
        name: param.clone()
        for name, param in model.state_dict().items()
    }

    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(
            train_loader
        ):
            if batch_idx >= num_batches:
                break

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(
                output,
                target
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(
            epoch_loss / num_batches
        )

    weight_updates = {
        name: param - initial_params[name]
        for name, param in model.state_dict().items()
    }

    return (
        weight_updates,
        sum(train_losses) / num_epochs
    )


def aggregate_weight_updates(
    updates_list
):
    aggregated_updates = {
        name: torch.zeros_like(
            updates_list[0][name]
        )
        for name in updates_list[0].keys()
    }

    for updates in updates_list:
        for name, update in updates.items():
            aggregated_updates[name] += update

    for name in aggregated_updates.keys():
        aggregated_updates[name] /= len(
            updates_list
        )

    return aggregated_updates


def update_model(
    model,
    aggregated_updates
):
    with torch.no_grad():
        for name, param in (
            model.state_dict().items()
        ):
            param += aggregated_updates[name]


def test_model(
    model,
    test_loader,
    criterion
):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(
                output,
                target
            )

            test_loss += (
                loss.item()
                * target.size(0)
            )

            pred = output.argmax(
                dim=1,
                keepdim=True
            )

            correct += pred.eq(
                target.view_as(pred)
            ).sum().item()

    test_loss /= len(
        test_loader.dataset
    )

    accuracy = (
        100.0
        * correct
        / len(test_loader.dataset)
    )

    return test_loss, accuracy


def client_train_1(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs,
    num_batches
):
    model.train()

    # 只训练一个 epoch 和一个 batch。
    data, target = next(
        iter(train_loader)
    )

    data = data.to(device)
    target = target.to(device)

    optimizer.zero_grad()

    output = model(data)

    selected_experts = (
        model.fc.last_selected_experts
    )

    if selected_experts is None:
        raise RuntimeError(
            'model.fc.last_selected_experts is None.'
        )

    expert_activation_frequency = (
        torch.bincount(
            selected_experts.reshape(-1),
            minlength=model.fc.num_experts
        ).to(
            device=device,
            dtype=torch.float32
        )
        / data.size(0)
    )

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

    model_params = tuple(
        model.params()
    )

    pseudo_grads = torch.autograd.grad(
        training_loss,
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

    # 只对专家参数按实际路由到该专家的样本数做内部平均。
    expert_routed_counts = torch.bincount(
        selected_experts.reshape(-1),
        minlength=model.fc.num_experts
    ).to(
        device=device,
        dtype=torch.float32
    )

    model_param_names = [
        name
        for name, _ in model.named_params(
            model
        )
    ]

    if len(model_param_names) != len(
        pseudo_grads
    ):
        raise RuntimeError(
            '参数名称数量与客户端梯度数量不一致'
        )

    routed_mean_pseudo_grads = []
    batch_sample_count = float(
        data.size(0)
    )

    for param_name, grad in zip(
        model_param_names,
        pseudo_grads
    ):
        if param_name.startswith(
            'fc.experts.'
        ):
            expert_id = int(
                param_name.split('.')[2]
            )

            routed_count = (
                expert_routed_counts[
                    expert_id
                ]
            )

            if routed_count.item() > 0:
                grad = grad * (
                    batch_sample_count
                    / routed_count
                )
            else:
                grad = torch.zeros_like(
                    grad
                )

        routed_mean_pseudo_grads.append(
            grad
        )

    pseudo_grads = tuple(
        routed_mean_pseudo_grads
    )

    return (
        cross_entropy_loss.detach(),
        load_balance_loss.detach(),
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
    default='false',
    help=(
        'Whether to use Dirichlet '
        'distribution for data splitting.'
    )
)

parser.add_argument(
    '--dirichlet_alpha',
    type=float,
    default=0.1,
    help=(
        'Alpha parameter for the '
        'Dirichlet distribution.'
    )
)

parser.add_argument(
    '--num_selected',
    type=int,
    default=100,
    help='Number of selected items.'
)

parser.add_argument(
    '--cache_client_train_loaders',
    type=str,
    default='true',
    choices=[
        'true',
        'false'
    ],
    help=(
        'Whether to reuse each client '
        'DataLoader across global rounds. '
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
    args.use_dirichlet.lower()
    == 'true'
)

dirichlet_alpha = (
    args.dirichlet_alpha
)

num_selected = (
    args.num_selected
)

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
log_start_time = (
    datetime.datetime.now()
)

log_time_str = (
    log_start_time.strftime(
        '%m%d_%H%M%S'
    )
)

os.makedirs(
    './log2',
    exist_ok=True
)

log_file_path = (
    f'./log2/Meta_LN_s_z_'
    f'{dataset}_'
    f'{log_time_str}.log'
)

original_stdout = sys.stdout

round_console_logger = (
    RoundOnlyConsoleLogger(
        log_file_path,
        original_stdout
    )
)

sys.stdout = round_console_logger

atexit.register(
    round_console_logger.close
)

print(
    'expert aggregation = mlp'
)

print(
    'nonexpert aggregation = fedavg'
)

print(
    'dataset = ',
    dataset
)


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
lr = 0.025
min_lr = 0.0001
decay_factor = 0.996

# Meta model parameters
meta_net_hidden_size = 500
meta_net_num_layers = 1
meta_lr = 1e-4
meta_weight_decay = 0

nesterov = True
momentum = 0.91
weight_decay = 5e-4

load_balance_coef = 0.0


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
        dirichlet_alpha=(
            dirichlet_alpha
        )
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
        dirichlet_alpha=(
            dirichlet_alpha
        )
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


# 初始化模型和优化器
global_model = build_model(
    dataset
)

num_experts = (
    global_model.fc.num_experts
)

criterion = nn.CrossEntropyLoss()

optimizer_model = torch.optim.SGD(
    global_model.params(),
    lr,
    momentum=momentum,
    nesterov=nesterov,
    weight_decay=weight_decay
)


client_model = (
    build_model(dataset)
    .to(device)
)


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


meta_dataloader_iter = iter(
    meta_dataloader
)


now = datetime.datetime.now()

time_str = now.strftime(
    '%m%d_%H%M'
)

best_acc = 0.0


experiment_config = [
    "EXPERIMENT_CONFIG_BEGIN",
    "method=meta_mlp_expert_aggregation",
    f"start_time={now.strftime('%Y-%m-%d_%H:%M:%S')}",
    f"log_file={log_file_path}",
    f"device={device}",
    f"dataset={dataset}",
    f"use_dirichlet={use_dirichlet}",
    f"dirichlet_alpha={dirichlet_alpha}",
    f"num_clients={num_clients}",
    f"clients_per_round={num_clients}",
    f"num_selected_arg={num_selected}",
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
    "expert_aggregation=meta_mlp_client_weighting",
    "nonexpert_aggregation=fedavg_equal_weight",
    (
        "meta_input="
        "standardized_client_cross_entropy_loss,"
        "expert_activation_frequency"
    ),
    (
        "loss_standardization="
        "zscore_per_round_across_clients"
    ),
    "client_loss_for_meta=cross_entropy_only",
    (
        "client_training_loss="
        "cross_entropy+load_balance"
    ),
    "meta_loss=cross_entropy_only",
    f"lr_max={lr}",
    f"lr_min={min_lr}",
    "lr_scheduler=cosine_annealing",
    f"momentum={momentum}",
    f"nesterov={nesterov}",
    f"weight_decay={weight_decay}",
    f"load_balance_coef={load_balance_coef}",
    f"meta_hidden_size={meta_net_hidden_size}",
    f"meta_num_layers={meta_net_num_layers}",
    "meta_optimizer=Adam",
    f"meta_lr={meta_lr}",
    f"meta_weight_decay={meta_weight_decay}",
    "EXPERIMENT_CONFIG_END",
]

print(
    "\n".join(
        experiment_config
    ),
    flush=True
)


fedavg_weight = torch.full(
    (num_clients,),
    1.0 / num_clients,
    dtype=torch.float32,
    device=device
)


for round in range(
    num_rounds
):
    current_lr = (
        min_lr
        + 0.5
        * (lr - min_lr)
        * (
            1.0
            + math.cos(
                math.pi
                * round
                / num_rounds
            )
        )
    )

    pseudo_net = build_model(
        dataset
    )

    pseudo_net.load_state_dict(
        global_model.state_dict()
    )

    client_losses = []
    client_expert_frequencies = []
    grads_list = []

    for i in range(
        num_clients
    ):
        client_model.load_state_dict(
            pseudo_net.state_dict()
        )

        train_loader = (
            get_client_train_loader(
                i
            )
        )

        client_optimizer = (
            torch.optim.SGD(
                client_model.params(),
                lr=current_lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay
            )
        )

        (
            loss,
            _load_balance_loss,
            expert_activation_frequency,
            weight_updates
        ) = client_train_1(
            client_model,
            train_loader,
            criterion,
            client_optimizer,
            num_epochs,
            num_batches
        )

        grads_list.append(
            weight_updates
        )

        client_losses.append(
            loss.item()
        )

        client_expert_frequencies.append(
            expert_activation_frequency
        )

        del client_optimizer


    client_losses_tensor = (
        torch.tensor(
            client_losses
        )
        .view(-1, 1)
        .to(device)
    )

    client_expert_frequencies_tensor = (
        torch.stack(
            client_expert_frequencies,
            dim=0
        )
        .to(device)
    )

    loss_mean = (
        client_losses_tensor.mean()
    )

    loss_std = (
        client_losses_tensor.std(
            unbiased=False
        )
        .clamp_min(1e-12)
    )

    standardized_client_losses_tensor = (
        client_losses_tensor
        - loss_mean
    ) / loss_std

    loss_features = (
        standardized_client_losses_tensor
        .unsqueeze(1)
        .expand(
            -1,
            num_experts,
            -1
        )
    )

    frequency_features = (
        client_expert_frequencies_tensor
        .unsqueeze(-1)
    )

    client_expert_features = (
        torch.cat(
            [
                loss_features,
                frequency_features
            ],
            dim=2
        )
    )

    raw_expert_weights = meta_net(
        client_expert_features.reshape(
            -1,
            2
        )
    ).view(
        num_clients,
        num_experts
    )

    expert_weights = (
        raw_expert_weights
        / raw_expert_weights.sum(
            dim=0,
            keepdim=True
        ).clamp_min(1e-12)
    )


    log_client_losses = (
        client_losses_tensor
        .detach()
        .cpu()
        .view(-1)
    )

    log_activation_frequencies = (
        client_expert_frequencies_tensor
        .detach()
        .cpu()
    )

    log_raw_weights = (
        raw_expert_weights
        .detach()
        .cpu()
    )

    log_normalized_weights = (
        expert_weights
        .detach()
        .cpu()
    )

    meta_weight_log_lines = []

    for client_id in range(
        num_clients
    ):
        for expert_id in range(
            num_experts
        ):
            meta_weight_log_lines.append(
                "META_WEIGHT_LOG "
                f"round_id={round + 1} "
                f"client_id={client_id} "
                f"expert_id={expert_id} "
                f"client_loss="
                f"{log_client_losses[client_id].item():.10f} "
                f"activation_frequency="
                f"{log_activation_frequencies[client_id, expert_id].item():.10f} "
                f"raw_weight="
                f"{log_raw_weights[client_id, expert_id].item():.10f} "
                f"normalized_weight="
                f"{log_normalized_weights[client_id, expert_id].item():.10f}"
            )

    print(
        "\n".join(
            meta_weight_log_lines
        ),
        flush=True
    )

    avg_client_loss = (
        client_losses_tensor
        .mean()
        .item()
    )


    param_names = [
        name
        for name, _ in (
            pseudo_net.named_params(
                pseudo_net
            )
        )
    ]

    if len(param_names) != len(
        grads_list[0]
    ):
        raise RuntimeError(
            "参数名称数量与客户端梯度数量不一致"
        )

    aggregated_grads = []

    for (
        param_index,
        param_name
    ) in enumerate(
        param_names
    ):
        if param_name.startswith(
            "fc.experts."
        ):
            expert_id = int(
                param_name.split('.')[2]
            )

            current_weights = (
                expert_weights[
                    :,
                    expert_id
                ]
            )
        else:
            current_weights = (
                fedavg_weight
            )

        aggregated_grad = (
            torch.zeros_like(
                grads_list[0][
                    param_index
                ]
            )
        )

        for client_id in range(
            num_clients
        ):
            aggregated_grad += (
                grads_list[
                    client_id
                ][param_index]
                * current_weights[
                    client_id
                ]
            )

        aggregated_grads.append(
            aggregated_grad
        )


    pseudo_net.update_params(
        lr_inner=current_lr,
        source_params=(
            aggregated_grads
        )
    )

    del aggregated_grads


    try:
        (
            meta_inputs,
            meta_labels
        ) = next(
            meta_dataloader_iter
        )
    except StopIteration:
        meta_dataloader_iter = iter(
            meta_dataloader
        )

        (
            meta_inputs,
            meta_labels
        ) = next(
            meta_dataloader_iter
        )

    meta_inputs = (
        meta_inputs.to(device)
    )

    meta_labels = (
        meta_labels.to(device)
    )

    meta_outputs = pseudo_net(
        meta_inputs
    )

    meta_loss = criterion(
        meta_outputs,
        meta_labels.long()
    )

    meta_optimizer.zero_grad()

    meta_loss.backward()

    meta_optimizer.step()

    global_model.load_state_dict(
        pseudo_net.state_dict()
    )


    (
        test_loss,
        test_accuracy
    ) = test_model(
        global_model,
        test_dataloader,
        criterion
    )

    best_acc = max(
        best_acc,
        test_accuracy
    )

    print(
        f"Round {round + 1} | "
        f"Client Loss: {avg_client_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.2f}% | "
        f"Best Acc: {best_acc:.2f}%",
        flush=True
    )
