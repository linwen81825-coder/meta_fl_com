import torch
import torch.nn as nn
from dataset.dataSplit_LN_new import get_data_loaders_new
from model.model import MLP
from model.wideresnet import SmallMetaConvNet, WideResNet, SmallMetaConvNet1 ,ResNet18
import datetime
from dataset.dataSplit_clothing1m import get_data_loaders_clothing1m
import argparse
import math
import os
import sys
import atexit


# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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

        # 完整内容始终写入日志。
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
        # 程序退出时，flush 可能在文件关闭后再次被调用
        if not self.log_stream.closed:
            self.log_stream.flush()

        try:
            self.console_stream.flush()
        except (ValueError, OSError):
            pass

    def close(self):
        # 避免重复关闭
        if self._closed:
            return

        self.flush()

        if not self.log_stream.closed:
            self.log_stream.close()

        self._closed = True



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



def copy_bn_running_stats(source_model, target_model):
    """
    Copy only BatchNorm running statistics from source_model to target_model.

    Important for this project: MetaConv/MetaLinear/BN affine weights are
    registered as buffers too, so we must NOT copy all named_buffers().
    Only running_mean / running_var / num_batches_tracked are non-learnable
    BN state that should follow the pseudo model's meta-batch forward.
    """
    source_buffers = dict(source_model.named_buffers())
    target_buffers = dict(target_model.named_buffers())

    bn_state_suffixes = (
        'running_mean',
        'running_var',
        'num_batches_tracked',
    )

    with torch.no_grad():
        for name, source_buffer in source_buffers.items():
            if not name.endswith(bn_state_suffixes):
                continue

            if name not in target_buffers:
                raise RuntimeError(
                    f'BN buffer {name} not found in target model.'
                )

            target_buffers[name].copy_(source_buffer)


def recompute_loss_after_pseudo_update(
    model,
    client_batches,
    criterion
):
    """
    Scheme A 的第二次输入：在 pseudo update 后，用每个客户端
    第一次训练时的同一批样本重新计算交叉熵 loss。

    使用 eval() + no_grad()，因此这一步不会再次训练模型，
    也不会额外修改 BatchNorm running statistics。
    """
    was_training = model.training
    model.eval()

    after_losses = []

    with torch.no_grad():
        for batch_data, batch_target in client_batches:
            batch_data = batch_data.to(
                device,
                non_blocking=True
            )
            batch_target = batch_target.to(
                device,
                non_blocking=True
            )

            output = model(batch_data)
            loss = criterion(
                output,
                batch_target
            )
            after_losses.append(loss.detach())

    if was_training:
        model.train()

    after_losses_tensor = torch.stack(
        after_losses,
        dim=0
    ).view(-1, 1)

    after_loss_mean = after_losses_tensor.mean()
    after_loss_std = after_losses_tensor.std(
        unbiased=False
    ).clamp_min(1e-12)

    standardized_after_losses_tensor = (
        after_losses_tensor - after_loss_mean
    ) / after_loss_std

    return (
        after_losses_tensor,
        standardized_after_losses_tensor
    )


def client_train_1(model, train_loader, criterion, optimizer, num_epochs, num_batches):
    model.train()

    # 只训练一个epoch和一个batch。
    # 同时保留这一批原始 CPU 数据，供 pseudo update 后重新评估
    # 第二次 MetaNet 输入；这样不会再次迭代 DataLoader。
    data, target = next(iter(train_loader))
    same_batch_data = data.detach().cpu()
    same_batch_target = target.detach().cpu()
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()

    output = model(data)

    # Top-2 专家激活频率。
    # last_selected_experts shape: [batch_size, top_k]
    selected_experts = (
        model.fc.last_selected_experts
    )

    if selected_experts is None:
        raise RuntimeError(
            'model.fc.last_selected_experts is None.'
        )

    # 每个样本产生 top_k 次专家分配。
    # 分母使用 batch_size * top_k，
    # 使所有专家激活频率之和保持为 1。
    expert_activation_frequency = (
        torch.bincount(
            selected_experts.reshape(-1),
            minlength=model.fc.num_experts
        ).to(
            device=device,
            dtype=torch.float32
        )
        / selected_experts.numel()
    )

    # 元网络输入仍然只使用交叉熵损失。
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

    # 负载均衡损失只参与模型梯度，
    # 不作为元网络输入，也不加入 meta_loss。
    training_loss = (
        cross_entropy_loss
        + load_balance_coef
        * load_balance_loss
    )

    model_params = tuple(model.params())
    # 计算权重更新量
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



    return (
        cross_entropy_loss.detach(),
        load_balance_loss.detach(),
        expert_activation_frequency.detach(),
        pseudo_grads,
        same_batch_data,
        same_batch_target
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
    default=100,
    help='Number of selected items.'
)

# 解析命令行参数
args = parser.parse_args()

use_dirichlet = args.use_dirichlet.lower() == 'true'
dirichlet_alpha = args.dirichlet_alpha
num_selected = args.num_selected
dataset = args.dataset

# 自动保存完整日志；控制台只显示每轮 Round 汇总行。
log_start_time = datetime.datetime.now()
log_time_str = log_start_time.strftime(
    '%m%d_%H%M%S'
)

os.makedirs(
    './log2',
    exist_ok=True
)

log_file_path = (
    f'./log2/Meta_LN_s_old_two_stage_meta_'
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
lr = 0.025
min_lr = 0.0001
decay_factor = 0.996

# Meta model parameters
meta_net_hidden_size = 500
meta_net_num_layers = 1
meta_lr = 1e-4
meta_weight_decay = 0

nesterov = True
momentum = 0.9
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
    input_size=1,
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

# 训练开始时只记录一次实验配置。
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
    f"meta_batch_size={meta_bs}",
    f"meta_sample_number={meta_sample_number}",
    f"model={global_model.__class__.__name__}",
    f"num_experts={num_experts}",
    "routing=top2",
    "top_k=2",
    "top2_output=renormalized_gate_weighted_sum",
    "expert_aggregation=meta_mlp_client_weighting",
    "nonexpert_aggregation=fedavg_equal_weight",
    "meta_input=standardized_client_cross_entropy_loss",
    "meta_input_size=1",
    "loss_standardization=zscore_per_round_across_clients",
    "expert_activation_frequency_used_by_meta=false",
    "client_loss_for_meta=cross_entropy_only",
    "client_training_loss=cross_entropy+load_balance",
    "meta_loss=cross_entropy_only",
    "meta_update_flow=pseudo_update_then_meta_step_then_reobserve_then_reweight",
    "meta_input_stage1=loss_before_z",
    "meta_input_stage2=loss_after_z",
    "stage2_batch=same_batch_as_stage1_client_training",
    "stage2_feature_eval=eval_no_grad",
    "global_update_uses=post_meta_second_weights",
    f"lr_max={lr}",
    f"lr_min={min_lr}",
    "lr_scheduler=cosine_annealing",
    f"momentum={momentum}",
    f"nesterov={nesterov}",
    f"weight_decay={weight_decay}",
    f"load_balance_coef={load_balance_coef}",
    f"meta_hidden_size={meta_net_hidden_size}",
    f"meta_num_layers={meta_net_num_layers}",
    f"meta_optimizer=Adam",
    f"meta_lr={meta_lr}",
    f"meta_weight_decay={meta_weight_decay}",
    "two_stage_bn_state=copy_from_pseudo_meta_forward",
    "EXPERIMENT_CONFIG_END",
]

print(
    "\n".join(experiment_config),
    flush=True
)

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
    client_same_batches = []

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
            _load_balance_loss,
            expert_activation_frequency,
            weight_updates,
            same_batch_data,
            same_batch_target
        ) = client_train_1(
            client_model,
            train_dataloaders[i],
            criterion,
            client_optimizer,
            num_epochs,
            num_batches
        )

        grads_list.append(weight_updates)

        # 元网络唯一输入是纯交叉熵 loss。
        client_losses.append(loss.item())

        client_expert_frequencies.append(
            expert_activation_frequency
        )

        client_same_batches.append(
            (
                same_batch_data,
                same_batch_target
            )
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

    # 对当前轮所有客户端的交叉熵 loss 做 Z-score 标准化。
    # 标准化只用于元网络输入，不改变客户端训练 loss、
    # 梯度计算、日志中的原始 client_loss 或其他训练流程。
    loss_mean = client_losses_tensor.mean()
    loss_std = client_losses_tensor.std(
        unbiased=False
    ).clamp_min(1e-12)

    standardized_client_losses_tensor = (
        client_losses_tensor - loss_mean
    ) / loss_std

    # ==================================================
    # 第一次 MetaNet 前向：只用于 pseudo update
    # ==================================================
    # 元网络输入只有标准化后的客户端交叉熵 loss：
    # [standardized_client_loss_k]
    #
    # 输入中没有专家激活频率或 expert_id，
    # 同一客户端对所有专家产生相同的原始权重。
    pseudo_raw_client_weights = meta_net(
        standardized_client_losses_tensor
    ).view(
        num_clients,
        1
    )

    pseudo_raw_expert_weights = (
        pseudo_raw_client_weights.expand(
            -1,
            num_experts
        )
    )
    # [num_clients, num_experts]

    # 每个专家分别沿客户端维度归一化。
    pseudo_expert_weights = (
        pseudo_raw_expert_weights
        / pseudo_raw_expert_weights.sum(
            dim=0,
            keepdim=True
        ).clamp_min(1e-12)
    )

    avg_client_loss = client_losses_tensor.mean().item()

    # 参数名称的顺序必须和 client_train_1() 中
    # model.params() 返回的梯度顺序一致。
    param_names = [
        name
        for name, _ in pseudo_net.named_params(pseudo_net)
    ]

    if len(param_names) != len(grads_list[0]):
        raise RuntimeError(
            "参数名称数量与客户端梯度数量不一致"
        )

    # ==================================================
    # 第一次聚合：使用更新前 MetaNet 的权重做 pseudo update
    # ==================================================
    pseudo_aggregated_grads = []

    for param_index, param_name in enumerate(
        param_names
    ):
        if param_name.startswith("fc.experts."):
            expert_id = int(
                param_name.split('.')[2]
            )
            current_weights = (
                pseudo_expert_weights[:, expert_id]
            )
        else:
            # 非专家参数始终使用等权 FedAvg。
            current_weights = fedavg_weight

        aggregated_grad = torch.zeros_like(
            grads_list[0][param_index]
        )

        for client_id in range(num_clients):
            aggregated_grad += (
                grads_list[client_id][param_index]
                * current_weights[client_id]
            )

        pseudo_aggregated_grads.append(
            aggregated_grad
        )

    # 这一步只产生用于 meta loss 的伪模型，
    # 不直接作为本轮最终 global_model。
    pseudo_net.update_params(
        lr_inner=current_lr,
        source_params=pseudo_aggregated_grads
    )

    # ==================================================
    # 用 pseudo_net 上的 meta loss 更新元学习网络
    # ==================================================
    del pseudo_aggregated_grads

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

    # ==================================================
    # Scheme A - Step 3: pseudo update 后重新观察客户端 loss
    # ==================================================
    # 第二次输入不再复用第一次的 loss_before_z。
    # 对每个客户端使用第一次训练时保存的同一批样本，
    # 在 pseudo_net 上重新计算 loss_after，并重新做跨客户端 Z-score。
    (
        client_losses_after_tensor,
        standardized_client_losses_after_tensor
    ) = recompute_loss_after_pseudo_update(
        pseudo_net,
        client_same_batches,
        criterion
    )

    # 保存的 batch 至此不再使用，尽早释放 CPU 引用。
    del client_same_batches

    # ==================================================
    # Scheme A - Step 4: 更新后的 MetaNet 使用 loss_after_z
    # ==================================================
    # 第二次权重不再参与 meta-gradient，因此关闭 autograd。
    with torch.no_grad():
        final_raw_client_weights = meta_net(
            standardized_client_losses_after_tensor
        ).view(
            num_clients,
            1
        )

        final_raw_expert_weights = (
            final_raw_client_weights.expand(
                -1,
                num_experts
            )
        )

        final_expert_weights = (
            final_raw_expert_weights
            / final_raw_expert_weights.sum(
                dim=0,
                keepdim=True
            ).clamp_min(1e-12)
        )

    # ==================================================
    # 第二次聚合：使用更新后 MetaNet 的权重做真实全局更新
    # ==================================================
    # 注意：这里从本轮尚未更新的 global_model 出发，
    # 而不是继续在 pseudo_net 上更新，避免同一轮梯度应用两次。
    final_aggregated_grads = []

    for param_index, param_name in enumerate(
        param_names
    ):
        if param_name.startswith("fc.experts."):
            expert_id = int(
                param_name.split('.')[2]
            )
            current_weights = (
                final_expert_weights[:, expert_id]
            )
        else:
            current_weights = fedavg_weight

        aggregated_grad = torch.zeros_like(
            grads_list[0][param_index]
        )

        for client_id in range(num_clients):
            aggregated_grad += (
                grads_list[client_id][param_index]
                * current_weights[client_id]
            )

        final_aggregated_grads.append(
            aggregated_grad
        )

    # global_model 在本轮至此尚未被更新，
    # 因此直接从原 global_model 应用第二次聚合梯度即可。
    # final update 不参与任何反向传播，在 no_grad 下执行，避免
    # 把无用计算图跨轮保存在 global_model 中。
    with torch.no_grad():
        global_model.update_params(
            lr_inner=current_lr,
            source_params=final_aggregated_grads
        )

    # 保留原代码的 BN 状态传播语义：meta batch 在 pseudo_net 上
    # 前向时会更新 running_mean/running_var。原来的
    # global_model.load_state_dict(pseudo_net.state_dict()) 会把这些统计量
    # 带回 global_model；两阶段更新时必须显式复制，否则测试时 BN 统计
    # 会长期停留在初始化附近。
    copy_bn_running_stats(
        source_model=pseudo_net,
        target_model=global_model
    )

    del final_aggregated_grads

    # ==================================================
    # 记录 MetaNet 更新前后两次权重的变化
    # ==================================================
    with torch.no_grad():
        normalized_weight_delta = (
            final_expert_weights
            - pseudo_expert_weights.detach()
        )
        mean_abs_weight_delta = (
            normalized_weight_delta.abs().mean().item()
        )
        max_abs_weight_delta = (
            normalized_weight_delta.abs().max().item()
        )

    with torch.no_grad():
        loss_z_delta = (
            standardized_client_losses_after_tensor
            - standardized_client_losses_tensor
        )
        mean_abs_loss_z_delta = (
            loss_z_delta.abs().mean().item()
        )

    log_client_losses = (
        client_losses_tensor.detach().cpu().view(-1)
    )
    log_client_losses_after = (
        client_losses_after_tensor.detach().cpu().view(-1)
    )
    log_loss_z_before = (
        standardized_client_losses_tensor.detach().cpu().view(-1)
    )
    log_loss_z_after = (
        standardized_client_losses_after_tensor.detach().cpu().view(-1)
    )
    log_activation_frequencies = (
        client_expert_frequencies_tensor.detach().cpu()
    )
    log_pseudo_raw_weights = (
        pseudo_raw_expert_weights.detach().cpu()
    )
    log_pseudo_normalized_weights = (
        pseudo_expert_weights.detach().cpu()
    )
    log_final_raw_weights = (
        final_raw_expert_weights.detach().cpu()
    )
    log_final_normalized_weights = (
        final_expert_weights.detach().cpu()
    )
    log_weight_delta = (
        normalized_weight_delta.detach().cpu()
    )

    meta_weight_log_lines = []

    for client_id in range(num_clients):
        for expert_id in range(num_experts):
            meta_weight_log_lines.append(
                "META_WEIGHT_LOG "
                f"round_id={round + 1} "
                f"client_id={client_id} "
                f"expert_id={expert_id} "
                f"client_loss_before={log_client_losses[client_id].item():.10f} "
                f"client_loss_after={log_client_losses_after[client_id].item():.10f} "
                f"loss_z_before={log_loss_z_before[client_id].item():.10f} "
                f"loss_z_after={log_loss_z_after[client_id].item():.10f} "
                f"activation_frequency="
                f"{log_activation_frequencies[client_id, expert_id].item():.10f} "
                f"pseudo_raw_weight="
                f"{log_pseudo_raw_weights[client_id, expert_id].item():.10f} "
                f"pseudo_normalized_weight="
                f"{log_pseudo_normalized_weights[client_id, expert_id].item():.10f} "
                f"final_raw_weight="
                f"{log_final_raw_weights[client_id, expert_id].item():.10f} "
                f"final_normalized_weight="
                f"{log_final_normalized_weights[client_id, expert_id].item():.10f} "
                f"normalized_weight_delta="
                f"{log_weight_delta[client_id, expert_id].item():.10f}"
            )

    print(
        "\n".join(meta_weight_log_lines),
        flush=True
    )

    print(
        "META_UPDATE_LOG "
        f"round_id={round + 1} "
        f"meta_loss={meta_loss.item():.10f} "
        f"mean_abs_loss_z_delta={mean_abs_loss_z_delta:.10f} "
        f"mean_abs_weight_delta={mean_abs_weight_delta:.10f} "
        f"max_abs_weight_delta={max_abs_weight_delta:.10f}",
        flush=True
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
        f"Meta Loss: {meta_loss.item():.4f} | "
        f"LossZ Δ: {mean_abs_loss_z_delta:.3e} | "
        f"Weight Δ: {mean_abs_weight_delta:.6e} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.2f}% | "
        f"Best Acc: {best_acc:.2f}%",
        flush=True
    )


    # torch.save(
    #     meta_net.state_dict(),
    #     (
    #         f'./save/mlp_model_s_LN'
    #         f'{num_selected}_N{num_clients}_'
    #         f'BS{batch_size}_{dataset}_'
    #         f'{time_str}.pth'
    #     )
    # )

    # # 两种模式都保存全局模型
    # torch.save(
    #     global_model.state_dict(),
    #     (
    #         f'./save/expert_mlp_nonexpert_fedavg_'
    #         f'global_model_s_LN'
    #         f'{num_selected}_N{num_clients}_'
    #         f'BS{batch_size}_{dataset}_'
    #         f'{time_str}.pth'
    #     )
    # )