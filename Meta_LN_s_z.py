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


def client_train_1(model, train_loader, criterion, optimizer, num_epochs, num_batches):
    model.train()

    # 只训练一个epoch和一个batch
    data, target = next(iter(train_loader))
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
        / data.size(0)
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

    # 只对专家参数按“实际路由到该专家的样本数”做内部平均。
    #
    # CrossEntropyLoss 默认对整个 batch 取 mean。
    # 当前专家反向不再乘 Gate 权重，因此专家 e 的原始梯度为：
    #
    #     g_e = (1 / B) * sum_{i routed to e} g_i
    #
    # 令 n_e 为当前 batch 中实际路由到专家 e 的样本数，
    # 对专家参数梯度乘 B / n_e 后得到：
    #
    #     g_e_mean = (1 / n_e) * sum_{i routed to e} g_i
    #
    # backbone、BN、Gate 等非专家参数完全不改。
    expert_routed_counts = torch.bincount(
        selected_experts.reshape(-1),
        minlength=model.fc.num_experts
    ).to(
        device=device,
        dtype=torch.float32
    )

    model_param_names = [
        name
        for name, _ in model.named_params(model)
    ]

    if len(model_param_names) != len(pseudo_grads):
        raise RuntimeError(
            '参数名称数量与客户端梯度数量不一致'
        )

    routed_mean_pseudo_grads = []
    batch_sample_count = float(data.size(0))

    for param_name, grad in zip(
        model_param_names,
        pseudo_grads
    ):
        if param_name.startswith('fc.experts.'):
            expert_id = int(
                param_name.split('.')[2]
            )

            routed_count = (
                expert_routed_counts[expert_id]
            )

            if routed_count.item() > 0:
                grad = grad * (
                    batch_sample_count
                    / routed_count
                )
            else:
                grad = torch.zeros_like(grad)

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
    f'./log2/Meta_LN_s_z_'
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
lr = 0.02
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
load_balance_coef = 0.01


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
    "meta_input=standardized_client_cross_entropy_loss,expert_activation_frequency",
    "loss_standardization=zscore_per_round_across_clients",
    "client_loss_for_meta=cross_entropy_only",
    "client_training_loss=cross_entropy+load_balance",
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
    f"meta_optimizer=Adam",
    f"meta_lr={meta_lr}",
    f"meta_weight_decay={meta_weight_decay}",
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

        # 元网络输入仍然是纯交叉熵 loss。
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

    # 对客户端 k、专家 e 构造：
    # [standardized_client_loss_k, activation_frequency_k_e]
    loss_features = (
        standardized_client_losses_tensor
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

    # 逐条记录元网络两个输入及对应输出权重。
    # round_id 使用 1-based；client_id 和 expert_id 使用 0-based。
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

    for client_id in range(num_clients):
        for expert_id in range(num_experts):
            meta_weight_log_lines.append(
                "META_WEIGHT_LOG "
                f"round_id={round + 1} "
                f"client_id={client_id} "
                f"expert_id={expert_id} "
                f"client_loss={log_client_losses[client_id].item():.10f} "
                f"activation_frequency="
                f"{log_activation_frequencies[client_id, expert_id].item():.10f} "
                f"raw_weight="
                f"{log_raw_weights[client_id, expert_id].item():.10f} "
                f"normalized_weight="
                f"{log_normalized_weights[client_id, expert_id].item():.10f}"
            )

    print(
        "\n".join(meta_weight_log_lines),
        flush=True
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

    # 两种模式都更新全局模型
    global_model.load_state_dict(
        pseudo_net.state_dict()
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