import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)
    
    #def replace_param(self,)
    
    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBasicBlock(MetaModule):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(MetaBasicBlock, self).__init__()

        self.bn1 = MetaBatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = MetaConv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and MetaConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class MetaNetworkBlock(MetaModule):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(MetaNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(MetaModule):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = MetaBasicBlock
        # 1st conv before any network block
        self.conv1 = MetaConv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = MetaNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = MetaNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = MetaNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = MetaBatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = MetaLinear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MetaLinear):
                m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class MetaMLPExpert(MetaModule):
    """
    单个 MLP 专家：
    feature -> hidden -> class logits
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes
    ):
        super(MetaMLPExpert, self).__init__()

        self.fc1 = MetaLinear(
            input_dim,
            hidden_dim
        )

        self.fc2 = MetaLinear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):
        x = F.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x


class MetaMoEHead(MetaModule):
    """
    Top-2 稀疏 MoE。

    Gate 为每个样本选择两个专家，只执行被选中的专家。
    两个专家的输出按照选中后的 Gate 概率重新归一化并加权求和。
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        num_experts=4,
        expert_hidden_dim=128,
        top_k=2
    ):
        super(MetaMoEHead, self).__init__()

        if top_k < 1:
            raise ValueError(
                f'top_k must be at least 1, got {top_k}.'
            )

        if top_k > num_experts:
            raise ValueError(
                f'top_k={top_k} cannot exceed '
                f'num_experts={num_experts}.'
            )

        self.num_experts = num_experts
        self.num_classes = num_classes
        self.top_k = top_k

        self.experts = nn.ModuleList([
            MetaMLPExpert(
                input_dim=input_dim,
                hidden_dim=expert_hidden_dim,
                num_classes=num_classes
            )
            for _ in range(num_experts)
        ])

        self.gate = MetaLinear(
            input_dim,
            num_experts
        )

        self.last_gate_weights = None

        # Top-2 专家索引，shape: [batch_size, top_k]。
        self.last_selected_experts = None

        # 保留旧字段作为兼容别名，表示概率最大的第一个专家。
        self.last_selected_expert = None

        self.last_load_balance_loss = None

    def forward(self, x):
        batch_size = x.size(0)

        # [B, num_experts]
        gate_logits = self.gate(x)

        gate_probs = torch.softmax(
            gate_logits,
            dim=1
        )

        # 每个样本选择概率最大的 top_k 个专家。
        # topk_probs / selected_experts:
        # [B, top_k]
        topk_probs, selected_experts = torch.topk(
            gate_probs,
            k=self.top_k,
            dim=1
        )

        # 只在被选中的专家之间重新归一化，
        # 保证每个样本的 Top-2 权重之和为 1。
        topk_weights = (
            topk_probs
            / topk_probs.sum(
                dim=1,
                keepdim=True
            ).clamp_min(1e-12)
        )

        # hard_gate[b, e] = 1 表示样本 b 的 Top-2
        # 路由结果中包含专家 e。
        hard_gate = torch.zeros_like(
            gate_probs
        )

        hard_gate.scatter_(
            dim=1,
            index=selected_experts,
            value=1.0
        )

        # Top-2 MoE 负载均衡辅助损失。
        #
        # 每个样本产生 top_k 次路由，因此除以 B * top_k，
        # 使所有专家的实际分配比例之和仍然等于 1。
        expert_fraction = (
            hard_gate.detach().sum(dim=0)
            / (
                batch_size
                * self.top_k
            )
        )

        router_probability = (
            gate_probs.mean(dim=0)
        )

        self.last_load_balance_loss = (
            self.num_experts
            * torch.sum(
                expert_fraction
                * router_probability
            )
        )

        output = x.new_zeros(
            batch_size,
            self.num_classes
        )

        # 只执行当前 batch 中实际被 Top-2 选中的专家。
        for expert_id, expert in enumerate(
            self.experts
        ):
            # 每一行是 [sample_index, topk_rank]。
            selected_positions = torch.nonzero(
                selected_experts == expert_id,
                as_tuple=False
            )

            if selected_positions.numel() == 0:
                continue

            sample_indices = (
                selected_positions[:, 0]
            )

            topk_rank_indices = (
                selected_positions[:, 1]
            )

            expert_inputs = x.index_select(
                0,
                sample_indices
            )

            expert_outputs = expert(
                expert_inputs
            )

            selected_weights = topk_weights[
                sample_indices,
                topk_rank_indices
            ].unsqueeze(1)

            # 前向仍按 Gate 权重加权：
            #     forward = expert_outputs * selected_weights
            #
            # 专家参数反向梯度不乘 Gate 权重：
            #     d(weighted_outputs) / d(expert_outputs) = 1
            #
            # Gate 仍可通过 selected_weights 接收分类损失梯度。
            weighted_outputs = (
                expert_outputs
                + expert_outputs.detach()
                * (
                    selected_weights
                    - 1.0
                )
            )

            output = output.index_add(
                0,
                sample_indices,
                weighted_outputs
            )

        self.last_gate_weights = (
            gate_probs.detach()
        )

        self.last_selected_experts = (
            selected_experts.detach()
        )

        # 兼容仍读取旧字段的代码。
        self.last_selected_expert = (
            selected_experts[:, 0].detach()
        )

        return output


class SmallMetaConvNet(MetaModule):
    def __init__(self, num_classes=10,num_experts=4,expert_hidden_dim=128):
        super(SmallMetaConvNet, self).__init__()

        # Define a simple sequential model using MetaConv2d and MetaLinear
        self.conv1 = MetaConv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = MetaBatchNorm2d(16)
        self.conv2 = MetaConv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = MetaBatchNorm2d(32)
        self.conv3 = MetaConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = MetaBatchNorm2d(64)
        self.feature_dim = 64 * 4 * 4
        self.fc = MetaMoEHead(input_dim=self.feature_dim, num_classes=num_classes,num_experts=num_experts,expert_hidden_dim=expert_hidden_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # Downsample by 2x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # Downsample by 2x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # Downsample by 2x
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

class SmallMetaConvNet1(MetaModule):
    def __init__(self, num_classes=10):
        super(SmallMetaConvNet1, self).__init__()
        self.conv1 = MetaConv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = MetaBatchNorm2d(16)
        self.conv2 = MetaConv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = MetaBatchNorm2d(32)
        self.conv3 = MetaConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = MetaBatchNorm2d(64)

        # Set the fully connected layer based on the expected flattened size
        self.fc = MetaLinear(64 * 7 * 7, num_classes)  # For 224x224 inputs

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x


class ResNet18(MetaModule):
    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=128,
        block=MetaBasicBlock,
        num_blocks=(2, 2, 2, 2)
    ):
        super(ResNet18, self).__init__()

        self.in_planes = 64

        # CIFAR-10/CIFAR-100 使用 3×3、stride=1，
        # 不使用 ImageNet ResNet 的 7×7 卷积和 maxpool。
        self.conv1 = MetaConv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = MetaBatchNorm2d(64)

        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1
        )

        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2
        )

        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2
        )

        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2
        )

        self.feature_dim = 512 * block.expansion

        # 必须命名为 self.fc。
        #
        # 训练代码依赖：
        # global_model.fc.num_experts
        # fc.experts.0.fc1.weight
        # fc.experts.1.fc2.bias
        self.fc = MetaMoEHead(
            input_dim=self.feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim
        )

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for current_stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    current_stride
                )
            )

            self.in_planes = (
                planes * block.expansion
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # [B, 512, H, W] -> [B, 512, 1, 1]
        out = F.adaptive_avg_pool2d(
            out,
            output_size=1
        )

        # [B, 512, 1, 1] -> [B, 512]
        out = torch.flatten(
            out,
            start_dim=1
        )

        # ResNet18特征送入MoE分类头。
        out = self.fc(out)

        return out

class ResNet(MetaModule):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet10():
    return ResNet(MetaBasicBlock, [1, 1, 1, 1])


class VNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)



    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return F.sigmoid(out)
class ResNet10Lite(MetaModule):
    """
    适合 CIFAR10 + Label Noise 的轻量 ResNet backbone

    结构:
        Conv3x3
        Layer1: 32
        Layer2: 64
        Layer3: 128
        GAP
        MetaMoEHead
    """

    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=128
    ):
        super(ResNet10Lite, self).__init__()

        self.in_planes = 32


        # CIFAR 输入
        self.conv1 = MetaConv2d(
            3,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = MetaBatchNorm2d(32)


        # 三个 residual stage
        self.layer1 = self._make_layer(
            MetaBasicBlock,
            32,
            blocks=1,
            stride=1
        )


        self.layer2 = self._make_layer(
            MetaBasicBlock,
            64,
            blocks=1,
            stride=2
        )


        self.layer3 = self._make_layer(
            MetaBasicBlock,
            128,
            blocks=1,
            stride=2
        )


        self.feature_dim = 128


        # 保持你的 MoE head 接口
        self.fc = MetaMoEHead(
            input_dim=self.feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim
        )



    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride
    ):

        layers = []

        strides = [stride] + [1] * (blocks-1)

        for s in strides:

            layers.append(
                block(
                    self.in_planes,
                    planes,
                    s
                )
            )

            self.in_planes = planes


        return nn.Sequential(*layers)



    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)


        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)


        # CIFAR global pooling
        x = F.adaptive_avg_pool2d(
            x,
            1
        )


        x = torch.flatten(
            x,
            1
        )


        x = self.fc(x)


        return x

class ResNet20Lite(MetaModule):

    """
    CIFAR10 ResNet20 backbone

    Conv
    Stage1: 3 blocks
    Stage2: 3 blocks
    Stage3: 3 blocks

    Feature:
        256 dim

    MoE Head:
        unchanged
    """

    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=128
    ):
        super(ResNet20Lite, self).__init__()

        self.in_planes = 16


        self.conv1 = MetaConv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = MetaBatchNorm2d(16)



        self.layer1 = self._make_layer(
            MetaBasicBlock,
            16,
            blocks=3,
            stride=1
        )


        self.layer2 = self._make_layer(
            MetaBasicBlock,
            32,
            blocks=3,
            stride=2
        )


        self.layer3 = self._make_layer(
            MetaBasicBlock,
            64,
            blocks=3,
            stride=2
        )


        # 提升feature维度
        self.feature_expand = MetaConv2d(
            64,
            256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.bn_last = MetaBatchNorm2d(256)



        self.feature_dim = 256


        self.fc = MetaMoEHead(
            input_dim=self.feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim
        )


    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride
    ):

        layers=[]

        strides=[
            stride
        ] + [
            1
        ]*(blocks-1)


        for s in strides:

            layers.append(
                block(
                    self.in_planes,
                    planes,
                    s
                )
            )

            self.in_planes = planes


        return nn.Sequential(*layers)



    def forward(self,x):

        x=self.conv1(x)
        x=self.bn1(x)
        x=F.relu(x)


        x=self.layer1(x)

        x=self.layer2(x)

        x=self.layer3(x)


        x=self.feature_expand(x)

        x=self.bn_last(x)

        x=F.relu(x)



        x=F.adaptive_avg_pool2d(
            x,
            1
        )


        x=torch.flatten(
            x,
            1
        )


        x=self.fc(x)

        return x
class MetaResBlock(MetaModule):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super(MetaResBlock,self).__init__()


        self.bn1 = MetaBatchNorm2d(
            in_channels
        )

        self.conv1 = MetaConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )


        self.bn2 = MetaBatchNorm2d(
            out_channels
        )

        self.conv2 = MetaConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )


        if in_channels != out_channels:

            self.shortcut = MetaConv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )

        else:

            self.shortcut = None



    def forward(self,x):

        identity=x


        out=F.relu(
            self.bn1(x)
        )

        out=self.conv1(out)


        out=F.relu(
            self.bn2(out)
        )

        out=self.conv2(out)


        if self.shortcut is not None:

            identity=self.shortcut(identity)


        out = out + identity


        return out
class MetaResNetConvNet(MetaModule):

    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=128
    ):

        super(
            MetaResNetConvNet,
            self
        ).__init__()



        self.conv1 = MetaConv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = MetaBatchNorm2d(
            16
        )


        self.block1 = MetaResBlock(
            16,
            16
        )


        self.block2 = MetaResBlock(
            16,
            32
        )


        self.block3 = MetaResBlock(
            32,
            64
        )


        self.feature_dim = 64*4*4



        self.fc = MetaMoEHead(
            input_dim=self.feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim
        )



    def forward(self,x):


        x=F.relu(
            self.bn1(
                self.conv1(x)
            )
        )


        x=self.block1(x)

        x=F.max_pool2d(
            x,
            2
        )


        x=self.block2(x)

        x=F.max_pool2d(
            x,
            2
        )


        x=self.block3(x)

        x=F.max_pool2d(
            x,
            2
        )


        x=x.view(
            x.size(0),
            -1
        )


        x=self.fc(x)


        return x
class MetaDepthwiseConv2d(MetaModule):

    def __init__(
        self,
        channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    ):
        super().__init__()

        conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride,
            padding,
            groups=channels,
            bias=bias
        )


        self.in_channels = channels
        self.out_channels = channels
        self.stride = conv.stride
        self.padding = conv.padding
        self.groups = channels
        self.kernel_size = conv.kernel_size


        self.register_buffer(
            'weight',
            to_var(
                conv.weight.data,
                requires_grad=True
            )
        )


        if bias:
            self.register_buffer(
                'bias',
                to_var(
                    conv.bias.data,
                    requires_grad=True
                )
            )

        else:
            self.register_buffer(
                'bias',
                None
            )


    def forward(self,x):

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            groups=self.groups
        )


    def named_leaves(self):

        return [
            ('weight',self.weight),
            ('bias',self.bias)
        ]
class MetaInvertedResidual(MetaModule):

    def __init__(
        self,
        inp,
        oup,
        stride,
        expand_ratio=4
    ):

        super().__init__()


        hidden_dim = inp * expand_ratio


        self.use_res_connect = (
            stride == 1
            and inp == oup
        )


        layers=[]


        # expansion
        if expand_ratio != 1:

            layers.extend([

                MetaConv2d(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),

                MetaBatchNorm2d(
                    hidden_dim
                ),

                nn.ReLU6(inplace=True)

            ])



        # depthwise

        layers.extend([

            MetaDepthwiseConv2d(
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1
            ),

            MetaBatchNorm2d(
                hidden_dim
            ),

            nn.ReLU6(inplace=True)


        ])



        # projection

        layers.extend([

            MetaConv2d(
                hidden_dim,
                oup,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),

            MetaBatchNorm2d(
                oup
            )

        ])


        self.conv = nn.Sequential(
            *layers
        )



    def forward(self,x):

        out=self.conv(x)


        if self.use_res_connect:

            return x+out

        else:

            return out
class MetaMobileNetV2Lite(MetaModule):


    """
    CIFAR10/CINIC10

    Lightweight MobileNetV2 backbone

    Conv
    MBConv
    MBConv
    MBConv

    GAP

    MetaMoEHead
    """


    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=128
    ):

        super().__init__()



        self.conv1 = nn.Sequential(

            MetaConv2d(
                3,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            MetaBatchNorm2d(
                32
            ),

            nn.ReLU6(inplace=True)

        )



        self.layer1 = MetaInvertedResidual(
            32,
            64,
            stride=1,
            expand_ratio=4
        )



        self.layer2 = nn.Sequential(

            MetaInvertedResidual(
                64,
                128,
                stride=2,
                expand_ratio=4
            ),

            MetaInvertedResidual(
                128,
                128,
                stride=1,
                expand_ratio=4
            )

        )



        self.layer3 = nn.Sequential(

            MetaInvertedResidual(
                128,
                256,
                stride=2,
                expand_ratio=4
            ),

            MetaInvertedResidual(
                256,
                256,
                stride=1,
                expand_ratio=4
            )

        )


        self.feature_dim=256



        self.fc=MetaMoEHead(

            input_dim=self.feature_dim,

            num_classes=num_classes,

            num_experts=num_experts,

            expert_hidden_dim=expert_hidden_dim

        )



    def forward(self,x):


        x=self.conv1(x)


        x=self.layer1(x)


        x=self.layer2(x)


        x=self.layer3(x)



        x=F.adaptive_avg_pool2d(
            x,
            1
        )


        x=torch.flatten(
            x,
            1
        )


        x=self.fc(x)


        return x
class MetaDenseLayer(MetaModule):

    """
    DenseNet bottleneck layer

    BN-ReLU-1x1 Conv
    BN-ReLU-3x3 Conv
    concat feature
    """

    def __init__(
        self,
        in_channels,
        growth_rate
    ):

        super(
            MetaDenseLayer,
            self
        ).__init__()


        self.bn1 = MetaBatchNorm2d(
            in_channels
        )

        self.conv1 = MetaConv2d(
            in_channels,
            growth_rate * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )


        self.bn2 = MetaBatchNorm2d(
            growth_rate * 4
        )


        self.conv2 = MetaConv2d(
            growth_rate * 4,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )


    def forward(self,x):

        out = F.relu(
            self.bn1(x)
        )

        out = self.conv1(out)


        out = F.relu(
            self.bn2(out)
        )

        out = self.conv2(out)


        # Dense connection
        out = torch.cat(
            [x,out],
            dim=1
        )

        return out
class MetaDenseBlock(MetaModule):

    def __init__(
        self,
        num_layers,
        in_channels,
        growth_rate
    ):

        super(
            MetaDenseBlock,
            self
        ).__init__()


        layers=[]


        channels=in_channels


        for i in range(num_layers):

            layer=MetaDenseLayer(
                channels,
                growth_rate
            )

            layers.append(layer)

            channels += growth_rate



        self.block=nn.Sequential(
            *layers
        )


        self.out_channels=channels



    def forward(self,x):

        return self.block(x)

class MetaTransition(MetaModule):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()


        self.bn=MetaBatchNorm2d(
            in_channels
        )


        self.conv=MetaConv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )


    def forward(self,x):

        x=F.relu(
            self.bn(x)
        )


        x=self.conv(x)


        x=F.avg_pool2d(
            x,
            2
        )

        return x

class MetaDenseNetLite(MetaModule):


    """
    CIFAR10/CINIC10 DenseNet backbone

    DenseBlock1
    DenseBlock2
    DenseBlock3

    feature_dim=256

    MetaMoEHead
    """

    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=128
    ):

        super(
            MetaDenseNetLite,
            self
        ).__init__()



        growth_rate=16



        self.conv1 = MetaConv2d(
            3,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )


        self.bn1 = MetaBatchNorm2d(
            32
        )



        # Dense Block 1

        self.block1 = MetaDenseBlock(
            num_layers=3,
            in_channels=32,
            growth_rate=growth_rate
        )


        channels1 = (
            32 + 3*growth_rate
        )


        self.trans1 = MetaTransition(
            channels1,
            64
        )



        # Dense Block 2

        self.block2 = MetaDenseBlock(
            num_layers=4,
            in_channels=64,
            growth_rate=growth_rate
        )


        channels2 = (
            64 + 4*growth_rate
        )


        self.trans2 = MetaTransition(
            channels2,
            128
        )



        # Dense Block 3

        self.block3 = MetaDenseBlock(
            num_layers=4,
            in_channels=128,
            growth_rate=growth_rate
        )


        channels3 = (
            128 + 4*growth_rate
        )


        # feature projection

        self.feature_expand = MetaConv2d(
            channels3,
            256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )


        self.bn_last = MetaBatchNorm2d(
            256
        )


        self.feature_dim=256



        self.fc = MetaMoEHead(
            input_dim=256,
            num_classes=num_classes,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim
        )



    def forward(self,x):


        x=self.conv1(x)

        x=self.bn1(x)

        x=F.relu(x)



        x=self.block1(x)

        x=self.trans1(x)



        x=self.block2(x)

        x=self.trans2(x)



        x=self.block3(x)



        x=self.feature_expand(x)

        x=self.bn_last(x)

        x=F.relu(x)



        x=F.adaptive_avg_pool2d(
            x,
            1
        )


        x=torch.flatten(
            x,
            1
        )


        x=self.fc(x)


        return x
class MetaDepthwiseConv2d(MetaModule):

    def __init__(
        self,
        channels,
        kernel_size=3,
        stride=1,
        padding=1
    ):
        super().__init__()

        conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride,
            padding,
            groups=channels,
            bias=False
        )


        self.stride = conv.stride
        self.padding = conv.padding
        self.groups = channels
        self.dilation = conv.dilation


        self.register_buffer(
            "weight",
            to_var(
                conv.weight.data,
                requires_grad=True
            )
        )


    def forward(self,x):

        return F.conv2d(
            x,
            self.weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


    def named_leaves(self):

        return [
            (
                "weight",
                self.weight
            )
        ]
class SmallMetaMBConv(MetaModule):

    def __init__(
        self,
        in_planes,
        out_planes,
        expand_ratio=2,
        stride=1
    ):

        super().__init__()


        hidden_dim = (
            in_planes * expand_ratio
        )


        self.use_res = (
            stride == 1
            and in_planes == out_planes
        )


        self.conv = nn.Sequential(

            # expand

            MetaConv2d(
                in_planes,
                hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),

            MetaBatchNorm2d(
                hidden_dim
            ),

            nn.ReLU6(inplace=True),



            # depthwise

            MetaDepthwiseConv2d(
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1
            ),

            MetaBatchNorm2d(
                hidden_dim
            ),

            nn.ReLU6(inplace=True),



            # projection

            MetaConv2d(
                hidden_dim,
                out_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),

            MetaBatchNorm2d(
                out_planes
            )

        )


    def forward(self,x):

        out=self.conv(x)


        if self.use_res:

            out=out+x


        return out
class SmallMetaMobileNetV2(MetaModule):

    """
    Small CNN replacement

    Structure:

    Conv16

    MBConv
    16 -> 32

    MBConv
    32 -> 64

    GAP

    MetaMoEHead

    """

    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=128
    ):

        super(
            SmallMetaMobileNetV2,
            self
        ).__init__()



        # stem

        self.conv1 = nn.Sequential(

            MetaConv2d(
                3,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            MetaBatchNorm2d(
                16
            ),

            nn.ReLU6(inplace=True)

        )



        # 对应原 conv2

        self.block1 = SmallMetaMBConv(

            16,
            32,

            expand_ratio=2,

            stride=2

        )



        # 对应原 conv3

        self.block2 = SmallMetaMBConv(

            32,
            64,

            expand_ratio=2,

            stride=2

        )



        self.feature_dim = 64



        self.fc = MetaMoEHead(

            input_dim=self.feature_dim,

            num_classes=num_classes,

            num_experts=num_experts,

            expert_hidden_dim=expert_hidden_dim

        )



    def forward(self,x):


        x=self.conv1(x)


        x=self.block1(x)


        x=self.block2(x)



        # CIFAR feature

        x=F.adaptive_avg_pool2d(
            x,
            1
        )


        x=torch.flatten(
            x,
            1
        )


        x=self.fc(x)


        return x
class SmallMetaDenseLayer(MetaModule):

    """
    Lightweight Dense layer

    BN-ReLU-Conv3x3
    concat feature

    """

    def __init__(
        self,
        in_channels,
        growth_rate=16
    ):

        super(
            SmallMetaDenseLayer,
            self
        ).__init__()


        self.bn = MetaBatchNorm2d(
            in_channels
        )


        self.conv = MetaConv2d(
            in_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )



    def forward(self,x):

        out = F.relu(
            self.bn(x)
        )


        out = self.conv(out)


        # Dense connection

        out = torch.cat(
            [
                x,
                out
            ],
            dim=1
        )


        return out
class SmallMetaDenseBlock(MetaModule):


    def __init__(
        self,
        in_channels,
        growth_rate=16,
        layers=1
    ):

        super(
            SmallMetaDenseBlock,
            self
        ).__init__()


        modules=[]


        channels=in_channels


        for i in range(layers):

            modules.append(
                SmallMetaDenseLayer(
                    channels,
                    growth_rate
                )
            )

            channels += growth_rate



        self.block=nn.Sequential(
            *modules
        )


        self.out_channels=channels



    def forward(self,x):

        return self.block(x)
class SmallMetaDenseNet(MetaModule):


    """
    Small Dense CNN for Federated Learning


    Structure:

    Conv16

    DenseBlock
    16 -> 32


    DenseBlock
    32 -> 48


    DenseBlock
    48 -> 64


    Flatten

    1024 feature

    MetaMoEHead


    """

    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=128
    ):

        super(
            SmallMetaDenseNet,
            self
        ).__init__()



        # stem

        self.conv1 = MetaConv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )


        self.bn1 = MetaBatchNorm2d(
            16
        )



        # block1

        self.block1 = SmallMetaDenseBlock(
            16,
            growth_rate=16
        )


        # 输出:
        # 16+16=32



        self.block2 = SmallMetaDenseBlock(
            32,
            growth_rate=16
        )


        # 输出:
        # 48



        self.block3 = SmallMetaDenseBlock(
            48,
            growth_rate=16
        )


        # 输出:
        # 64



        self.pool = nn.MaxPool2d(
            2
        )



        self.feature_dim = 64*4*4



        self.fc = MetaMoEHead(

            input_dim=self.feature_dim,

            num_classes=num_classes,

            num_experts=num_experts,

            expert_hidden_dim=expert_hidden_dim

        )



    def forward(self,x):


        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )


        # 32×32

        x=self.block1(x)

        x=self.pool(x)


        # 16×16

        x=self.block2(x)

        x=self.pool(x)


        # 8×8

        x=self.block3(x)

        x=self.pool(x)


        # 4×4

        x=x.view(
            x.size(0),
            -1
        )


        x=self.fc(x)


        return x
class MetaSEBlock(MetaModule):

    def __init__(
        self,
        channels,
        reduction=4
    ):
        super().__init__()


        self.pool = nn.AdaptiveAvgPool2d(1)


        self.fc1 = MetaConv2d(
            channels,
            channels // reduction,
            kernel_size=1
        )


        self.fc2 = MetaConv2d(
            channels // reduction,
            channels,
            kernel_size=1
        )


    def forward(self,x):

        w = self.pool(x)


        w = F.relu(
            self.fc1(w)
        )


        w = torch.sigmoid(
            self.fc2(w)
        )


        return x*w
class SmallMetaEfficientBlock(MetaModule):

    """
    EfficientNet-lite MBConv

    Expand
    Depthwise
    SE
    Project
    """

    def __init__(
        self,
        in_planes,
        out_planes,
        expand_ratio=2,
        stride=1
    ):

        super().__init__()


        hidden_dim = (
            in_planes * expand_ratio
        )


        self.use_res = (
            stride==1
            and in_planes==out_planes
        )


        self.expand = nn.Sequential(

            MetaConv2d(
                in_planes,
                hidden_dim,
                kernel_size=1,
                bias=False
            ),

            MetaBatchNorm2d(
                hidden_dim
            ),

            nn.SiLU(inplace=True)

        )


        self.depthwise = nn.Sequential(

            MetaDepthwiseConv2d(
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1
            ),

            MetaBatchNorm2d(
                hidden_dim
            ),

            nn.SiLU(inplace=True)

        )


        self.se = MetaSEBlock(
            hidden_dim
        )


        self.project = nn.Sequential(

            MetaConv2d(
                hidden_dim,
                out_planes,
                kernel_size=1,
                bias=False
            ),

            MetaBatchNorm2d(
                out_planes
            )

        )



    def forward(self,x):

        identity=x


        out=self.expand(x)

        out=self.depthwise(out)


        out=self.se(out)


        out=self.project(out)


        if self.use_res:

            out=out+identity


        return out
class SmallMetaEfficientNet(MetaModule):

    """
    CIFAR EfficientNet Lite


    Conv16

    EfficientBlock
    16 -> 32


    EfficientBlock
    32 -> 64


    Flatten

    1024 feature

    MetaMoEHead

    """

    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=128
    ):

        super(
            SmallMetaEfficientNet,
            self
        ).__init__()



        self.conv1 = nn.Sequential(

            MetaConv2d(
                3,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            MetaBatchNorm2d(
                16
            ),

            nn.SiLU(inplace=True)

        )



        self.block1 = SmallMetaEfficientBlock(

            16,
            32,

            expand_ratio=2,

            stride=2

        )



        self.block2 = SmallMetaEfficientBlock(

            32,
            64,

            expand_ratio=2,

            stride=2

        )


        self.pool = nn.MaxPool2d(
            2
        )


        self.feature_dim = (
            64*4*4
        )



        self.fc = MetaMoEHead(

            input_dim=self.feature_dim,

            num_classes=num_classes,

            num_experts=num_experts,

            expert_hidden_dim=expert_hidden_dim

        )



    def forward(self,x):


        x=self.conv1(x)



        # 32×32

        x=self.block1(x)


        # 16×16

        x=self.block2(x)


        # 8×8

        x=self.pool(x)


        # 4×4


        x=x.view(
            x.size(0),
            -1
        )


        x=self.fc(x)


        return x
class SmallMetaVGGBlock(MetaModule):

    """
    VGG style block

    Conv3x3
    BN
    ReLU

    Conv3x3
    BN
    ReLU

    """

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super(
            SmallMetaVGGBlock,
            self
        ).__init__()


        self.conv1 = MetaConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = MetaBatchNorm2d(
            out_channels
        )


        self.conv2 = MetaConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = MetaBatchNorm2d(
            out_channels
        )



    def forward(self,x):

        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )


        x = F.relu(
            self.bn2(
                self.conv2(x)
            )
        )


        return x
class SmallMetaVGG(MetaModule):

    """
    Lightweight VGG backbone


    Stage1:
        3 -> 16 -> 16


    Stage2:
        16 -> 32 -> 32


    Stage3:
        32 -> 64 -> 64


    Feature:
        1024


    Head:
        MetaMoEHead

    """

    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=128
    ):

        super(
            SmallMetaVGG,
            self
        ).__init__()



        self.stage1 = SmallMetaVGGBlock(
            3,
            16
        )


        self.stage2 = SmallMetaVGGBlock(
            16,
            32
        )


        self.stage3 = SmallMetaVGGBlock(
            32,
            64
        )


        self.pool = nn.MaxPool2d(
            kernel_size=2
        )



        # 保持和 SmallMetaConvNet 一致

        self.feature_dim = (
            64*4*4
        )


        self.fc = MetaMoEHead(
            input_dim=self.feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim
        )



    def forward(self,x):


        # 32×32

        x = self.stage1(x)

        x = self.pool(x)



        # 16×16

        x = self.stage2(x)

        x = self.pool(x)



        # 8×8

        x = self.stage3(x)

        x = self.pool(x)



        # 4×4

        x = x.view(
            x.size(0),
            -1
        )


        x = self.fc(x)


        return x
class SmallMetaConvNet64(MetaModule):
    """
    Tiny-ImageNet 64x64 version

    Structure:
        Conv16
        Conv32
        Conv64

        3 times downsample

        Adaptive pooling

        MetaMoEHead

    Input:
        3 x 64 x 64

    Feature:
        1024
    """

    def __init__(
        self,
        num_classes=200,
        num_experts=4,
        expert_hidden_dim=128
    ):
        super(
            SmallMetaConvNet64,
            self
        ).__init__()


        self.conv1 = MetaConv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn1 = MetaBatchNorm2d(16)



        self.conv2 = MetaConv2d(
            16,
            32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn2 = MetaBatchNorm2d(32)



        self.conv3 = MetaConv2d(
            32,
            64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn3 = MetaBatchNorm2d(64)



        # 关键：
        # 无论输入64×64，
        # 最终固定成4×4
        self.pool = nn.AdaptiveAvgPool2d(
            (4,4)
        )


        self.feature_dim = (
            64 * 4 * 4
        )


        self.fc = MetaMoEHead(
            input_dim=self.feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim
        )



    def forward(self,x):

        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = F.max_pool2d(
            x,
            2
        )


        x = F.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        x = F.max_pool2d(
            x,
            2
        )


        x = F.relu(
            self.bn3(
                self.conv3(x)
            )
        )

        x = F.max_pool2d(
            x,
            2
        )


        # 64输入:
        # 64 -> 32 -> 16 -> 8
        #
        # Adaptive:
        # 8×8 -> 4×4

        x = self.pool(x)


        x = torch.flatten(
            x,
            1
        )


        x = self.fc(x)


        return x
class SmallMetaConvNet96(MetaModule):
    """
    STL-10 96x96 version

    Input:
        3 x 96 x 96

    Structure:
        Conv32
        Conv64
        Conv128
        AdaptivePool
        MetaMoEHead

    Feature:
        2048
    """

    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=128
    ):
        super(
            SmallMetaConvNet96,
            self
        ).__init__()


        self.conv1 = MetaConv2d(
            3,
            32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn1 = MetaBatchNorm2d(32)



        self.conv2 = MetaConv2d(
            32,
            64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn2 = MetaBatchNorm2d(64)



        self.conv3 = MetaConv2d(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn3 = MetaBatchNorm2d(128)



        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # 12×12 -> 4×4
        self.avgpool = nn.AdaptiveAvgPool2d(
            (4,4)
        )


        self.feature_dim = 128 * 4 * 4


        self.fc = MetaMoEHead(
            input_dim=self.feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim
        )



    def forward(self, x):

        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = self.pool(x)


        x = F.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        x = self.pool(x)


        x = F.relu(
            self.bn3(
                self.conv3(x)
            )
        )

        x = self.pool(x)


        x = self.avgpool(x)


        x = torch.flatten(
            x,
            1
        )


        x = self.fc(x)


        return x