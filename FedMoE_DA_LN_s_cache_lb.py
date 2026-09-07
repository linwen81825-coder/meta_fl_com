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
    ResNet10Lite,
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






def build_model(dataset, layers=10, widen_factor=1, droprate=0):
    if dataset in ['cifar10','cinic10','svhn']:
        model = SmallMetaVGG(num_classes=10)
        # model = ResNet10Lite(num_classes=10,num_experts=4,expert_hidden_dim=128)
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

    expert_usage = (
        torch.bincount(
            model.fc.last_selected_experts.reshape(-1),
            minlength=model.fc.num_experts
        )
        .cpu()
        .float()
    )

    return (
        train_loss / trained_batches,
        weight_updates,
        expert_usage
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




class AggregationEvaluation:
    """Read-only aggregation probes: restore module modes and all RNG states."""

    def __init__(self, module):
        self.module = module

    def __enter__(self):
        self.modes = [(module, module.training) for module in self.module.modules()]
        self.torch_state = torch.get_rng_state()
        self.cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.numpy_state = np.random.get_state()
        self.python_state = random.getstate()
        self.module.eval()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, training in self.modes:
            module.training = training
        torch.set_rng_state(self.torch_state)
        if self.cuda_states is not None:
            torch.cuda.set_rng_state_all(self.cuda_states)
        np.random.set_state(self.numpy_state)
        random.setstate(self.python_state)
        return False


class PaperExpertAggregator:
    """
    两篇论文的“仅专家聚合”适配；输出仍是原脚本的单个全局 MoE。

    FedMoE-DA: https://arxiv.org/html/2411.02115v2, Eq. (8)-(11).
      1. 将所有客户端、所有位置的专家及其 gate 代理向量放入池中。
      2. 每个客户端专家按代理向量 cosine similarity 选择 P 个邻居，
         连同自身做 masked softmax / tau 加权的参数聚合。
      3. 原文得到个性化专家；本适配对各客户端同一位置的聚合结果再
         等权平均，得到当前框架所需的全局专家。允许跨位置匹配。
      每轮重算相关矩阵，即原文聚合间隔 I=1；不引入 P2P 训练流程。

    Fed-MoE (IJCAI 2025): https://www.ijcai.org/proceedings/2025/0610.pdf
      移植 Eq. (3)-(6) 中 routed experts 的聚合部分：
          Q[x,e] = server gate probability
          P_y[x,j] = probability of the true reference label from source expert j
          W = mean_x(Q[x,:] outer P_y[x,:])
          A = row_softmax(W)
          E_new = (1-lambda) * E_old + lambda * A @ local_expert_parameters
      上式明确采用“先对参考样本的外积取平均，再 row-softmax”的批量约定。
      一轮通信做一次该移动平均；lambda 默认 0.5 是本适配的可调默认值。
      使用数据加载器原已从训练集划出的 metadata，不使用 test_loader。
      参考样本由本轮起点的全局 backbone 提取公共特征，然后分别送入
      各客户端 MLP 专家，避免将客户端 backbone 差异混入专家相关性。
      服务器 Q 使用现有全局 gate；不增加 main expert、gate 优化器或
      原文 Stage-C 的个性化同步，以保持原模型及非专家训练流程。

    两种适配均聚合完整专家参数，再减去目标位置的旧全局参数返回 delta。
    跨位置聚合时不能直接混合 local delta，因为不同位置的初始专家不同。
    """

    def __init__(self, method, model, reference_loader, options):
        if method not in ('fedmoe', 'fedmoe_da'):
            raise ValueError(f'Unknown expert aggregation: {method}')
        self.method = method
        self.num_experts = int(model.fc.num_experts)
        self.moving_rate = float(options.fedmoe_moving_rate)
        self.neighbors = int(options.fedmoeda_neighbors)
        self.temperature = float(options.fedmoeda_temperature)
        self.reference_batches = int(options.fedmoe_reference_batches)
        if not math.isfinite(self.moving_rate) or not 0.0 <= self.moving_rate <= 1.0:
            raise ValueError('--fedmoe_moving_rate must be between 0 and 1.')
        if self.neighbors < 0:
            raise ValueError('--fedmoeda_neighbors must be nonnegative.')
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError('--fedmoeda_temperature must be positive.')
        if self.reference_batches < 0:
            raise ValueError('--fedmoe_reference_batches must be nonnegative.')
        self.source_experts = []
        self.source_proxies = []
        self.source_confidences = []
        self.client_count = 0
        if self.method == 'fedmoe':
            self._prepare_reference(model, reference_loader)

    @torch.no_grad()
    def _prepare_reference(self, model, reference_loader):
        if reference_loader is None or len(reference_loader.dataset) == 0:
            raise ValueError('Fed-MoE expert aggregation needs a nonempty held-out metadata loader.')
        # Dedicated iterator; never iterate a client loader or the test loader.
        generator = torch.Generator().manual_seed(90210)
        loader = DataLoader(
            reference_loader.dataset,
            batch_size=reference_loader.batch_size or 128,
            shuffle=False,
            num_workers=0,
            generator=generator,
        )
        feature_batches, label_batches, gate_batches = [], [], []
        head = model.fc
        model_device = head.gate.weight.device
        with AggregationEvaluation(model):
            # Temporary identity exposes backbone features for CNN/VGG/ResNet
            # without changing the repository's model source or running MoE.
            model.fc = nn.Identity()
            try:
                for batch_idx, (data, target) in enumerate(loader):
                    if self.reference_batches and batch_idx >= self.reference_batches:
                        break
                    features = model(data.to(model_device, non_blocking=True))
                    if features.ndim != 2:
                        raise ValueError('Expected [batch, feature_dim] input to model.fc.')
                    probabilities = torch.softmax(head.gate(features), dim=1)
                    feature_batches.append(features.detach().cpu().clone())
                    label_batches.append(target.detach().cpu().long().clone())
                    gate_batches.append(probabilities.detach().cpu().float().clone())
            finally:
                model.fc = head
        self.reference_features = feature_batches
        self.reference_labels = label_batches
        self.server_gate_probabilities = torch.cat(gate_batches, dim=0)

    @torch.no_grad()
    def collect(self, local_model, weight_updates, reference_state):
        """Keep expert tensors on CPU, in client-major / expert-major order."""
        if int(local_model.fc.num_experts) != self.num_experts:
            raise ValueError('All clients must use the existing common expert count.')
        for expert_id in range(self.num_experts):
            prefix = f'fc.experts.{expert_id}.'
            expert_state = {
                name[len(prefix):]: reference_state[name] + update
                for name, update in weight_updates.items()
                if name.startswith(prefix)
            }
            if not expert_state:
                raise ValueError(f'Missing expert parameters: {prefix}')
            if self.source_experts:
                prototype = self.source_experts[0]
                if expert_state.keys() != prototype.keys() or any(
                    expert_state[name].shape != prototype[name].shape for name in prototype
                ):
                    raise ValueError('Expert aggregation requires matching expert architectures.')
            self.source_experts.append(expert_state)

        if self.method == 'fedmoe_da':
            # Paper proxies are columns of Pi; PyTorch Linear stores them as rows.
            # Use the gate weight vectors from Eq. (8), not routing counts/bias.
            proxies = local_model.fc.gate.weight.detach().cpu().float().clone()
            if proxies.ndim != 2 or proxies.shape[0] != self.num_experts:
                raise ValueError('Expected a linear gate with weight shape [experts, features].')
            self.source_proxies.append(proxies)
        else:
            confidence_batches = []
            model_device = local_model.fc.gate.weight.device
            with AggregationEvaluation(local_model.fc):
                for features_cpu, labels_cpu in zip(self.reference_features, self.reference_labels):
                    features = features_cpu.to(model_device, non_blocking=True)
                    labels = labels_cpu.to(model_device, non_blocking=True)
                    confidences = []
                    for expert in local_model.fc.experts:
                        probabilities = torch.softmax(expert(features), dim=1)
                        confidences.append(probabilities.gather(1, labels[:, None]).squeeze(1))
                    confidence_batches.append(torch.stack(confidences, dim=1).cpu().float())
            self.source_confidences.append(torch.cat(confidence_batches, dim=0))
        self.client_count += 1

    @torch.no_grad()
    def aggregation_weights(self):
        if not self.client_count:
            raise RuntimeError('No client expert states were collected.')
        if self.method == 'fedmoe_da':
            proxies = torch.cat(self.source_proxies, dim=0)
            proxies = torch.nn.functional.normalize(proxies, p=2, dim=1, eps=1e-12)
            similarities = (proxies @ proxies.t()).clamp(-1.0, 1.0)
            if not torch.isfinite(similarities).all():
                raise RuntimeError('Non-finite FedMoE-DA gate similarity.')
            source_count = similarities.size(0)
            if self.neighbors == 0:
                # Explicit no-neighbor ablation: keep each local expert itself.
                attention = torch.eye(source_count, dtype=similarities.dtype)
            else:
                count = min(self.neighbors + 1, source_count)
                threshold = similarities.topk(count, dim=1).values[:, -1:]
                # Eq. (9) uses >=; ties can select more than P+1 experts.
                selected = similarities >= threshold
                selected.fill_diagonal_(True)
                scores = (similarities / self.temperature).masked_fill(~selected, float('-inf'))
                attention = torch.softmax(scores, dim=1)
            # Eq. (11), then the explicit projection to one global expert/slot.
            return attention.reshape(
                self.client_count, self.num_experts, source_count
            ).mean(dim=0)

        confidences = torch.cat(self.source_confidences, dim=1)
        gate = self.server_gate_probabilities
        if confidences.shape[0] != gate.shape[0] or gate.shape[1] != self.num_experts:
            raise RuntimeError('Reference response dimensions do not match.')
        correlation = gate.t() @ confidences / gate.shape[0]
        if not torch.isfinite(correlation).all():
            raise RuntimeError('Non-finite Fed-MoE reference correlation.')
        return torch.softmax(correlation, dim=1)

    @torch.no_grad()
    def aggregate(self, reference_state):
        weights = self.aggregation_weights()
        expert_updates = {}
        for suffix in self.source_experts[0]:
            # Materialize one parameter field at a time, not 100 full models.
            values = torch.stack([state[suffix] for state in self.source_experts])
            work_dtype = values.dtype if values.is_floating_point() or values.is_complex() else torch.float32
            if work_dtype in (torch.float16, torch.bfloat16):
                work_dtype = torch.float32
            mixed = weights.to(work_dtype) @ values.to(work_dtype).reshape(len(self.source_experts), -1)
            mixed = mixed.reshape(self.num_experts, *values.shape[1:])
            for expert_id in range(self.num_experts):
                name = f'fc.experts.{expert_id}.{suffix}'
                reference = reference_state[name]
                new_value = mixed[expert_id]
                if self.method == 'fedmoe':
                    new_value = (1.0 - self.moving_rate) * reference.to(work_dtype) + self.moving_rate * new_value
                if not reference.is_floating_point() and not reference.is_complex():
                    new_value = new_value.round()
                expert_updates[name] = new_value.to(reference.dtype) - reference
        return expert_updates


def average_weight_updates(
    update_sums,
    reference_state,
    num_clients,
    expert_updates=None
):
    """Only expert deltas are replaced; all other averaging is unchanged."""
    aggregated_updates = {}
    for name, update_sum in update_sums.items():
        reference = reference_state[name]
        if expert_updates is not None and name.startswith('fc.experts.'):
            aggregated_update = expert_updates[name]
        elif reference.is_floating_point() or reference.is_complex():
            aggregated_update = update_sum / num_clients
        else:
            aggregated_update = (update_sum / num_clients).round()
        aggregated_updates[name] = aggregated_update.to(reference.dtype)
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
    default='false'
)

parser.add_argument(
    '--dirichlet_alpha',
    type=float,
    default=0.1
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

parser.add_argument(
    '--aggregation',
    type=str,
    default='fedmoe_da',
    choices=['fedmoe','fedmoe_da']
)

# Expert-aggregation-only options; original training defaults are unchanged.
parser.add_argument(
    '--fedmoe_moving_rate', type=float, default=0.5,
    help='Fed-MoE Eq.(6) mixing rate; 0.5 is the expert-only adaptation default.'
)
parser.add_argument(
    '--fedmoe_reference_batches', type=int, default=0,
    help='Reference metadata batches per round; 0 uses the full existing metadata set.'
)
parser.add_argument(
    '--fedmoeda_neighbors', type=int, default=5,
    help='FedMoE-DA P: related experts in addition to self; ties follow Eq.(9).'
)
parser.add_argument(
    '--fedmoeda_temperature', type=float, default=1.0,
    help='FedMoE-DA tau in Eq.(10).'
)

args = parser.parse_args()

use_dirichlet = (
    args.use_dirichlet.lower() == 'true'
)

dirichlet_alpha = args.dirichlet_alpha
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
    './log_lb',
    exist_ok=True
)

log_file_path = (
    f'./log_lb/FedMoE_DA_LN_s_cache_'
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
momentum = 0.2
weight_decay = 5e-4

# Top-2 MoE 负载均衡辅助损失系数。
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
    "method=fedmoe_da",
    f"start_time={now.strftime('%Y-%m-%d_%H:%M:%S')}",
    f"log_file={log_file_path}",
    f"device={device}",
    f"dataset={dataset}",
    f"use_dirichlet={use_dirichlet}",
    f"dirichlet_alpha={dirichlet_alpha}",
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
    f"aggregation={args.aggregation}_expert_only_paper_adaptation",
    "nonexpert_aggregation=equal_average_full_client_updates",
    "expert_source_pool=all_client_experts_all_slots",
    f"fedmoe_moving_rate={args.fedmoe_moving_rate}",
    f"fedmoe_reference_batches={args.fedmoe_reference_batches}",
    "fedmoe_reference_source=existing_training_metadata",
    "fedmoe_correlation=mean_reference_outer_product_then_row_softmax",
    "fedmoe_server_gate=current_global_gate_read_only",
    f"fedmoeda_neighbors={args.fedmoeda_neighbors}",
    f"fedmoeda_temperature={args.fedmoeda_temperature}",
    "fedmoeda_global_projection=mean_client_aggregates_per_slot",
    "fedmoeda_aggregation_interval=1",
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

    # Reference probing is confined to aggregation and preserves model/RNG state.
    paper_aggregator = PaperExpertAggregator(
        args.aggregation, global_model, meta_dataloader, args
    )

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
            weight_updates,
            expert_usage
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

        # Collect full expert states plus paper-specific aggregation information.
        paper_aggregator.collect(
            local_model, weight_updates, global_state_cpu
        )

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

    expert_updates = paper_aggregator.aggregate(global_state_cpu)
    del paper_aggregator

    aggregated_updates = (
        average_weight_updates(
            update_sums,
            global_state_cpu,
            num_clients,
            expert_updates=expert_updates
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
