import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 定义简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# 定义客户端训练函数
def client_train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    initial_params = {name: param.clone() for name, param in model.state_dict().items()}

    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # 计算权重更新量
    weight_updates = {name: param - initial_params[name] for name, param in model.state_dict().items()}
    return weight_updates


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


# 示例数据
data = torch.randn(100, 10)
targets = torch.randn(100, 1)
dataset = TensorDataset(data, targets)
train_loader = DataLoader(dataset, batch_size=10)

# 初始化模型和优化器
global_model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(global_model.parameters(), lr=0.01)

# 模拟多个客户端
num_clients = 5
client_models = [SimpleModel() for _ in range(num_clients)]
client_optimizers = [optim.SGD(model.parameters(), lr=0.01) for model in client_models]

# 将全局模型参数分发给每个客户端
for client_model in client_models:
    client_model.load_state_dict(global_model.state_dict())

# 客户端训练并上传权重更新
num_epochs = 5  # 客户端本地训练的epoch数
all_weight_updates = []
for i in range(num_clients):
    weight_updates = client_train(client_models[i], train_loader, criterion, client_optimizers[i], num_epochs)
    all_weight_updates.append(weight_updates)

# 聚合权重更新并更新全局模型
aggregated_weight_updates = aggregate_weight_updates(all_weight_updates)
update_model(global_model, aggregated_weight_updates)

print("模型更新完成")
