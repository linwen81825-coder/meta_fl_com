import scipy.io as sio

# modules
from torchviz import make_dot

from sampling import *
from dataset.dataSplit_LN_new import get_data_loaders_new
from model_operation import *
from model import ResNet32, MLP


# GPU settings
torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# FL data setting
NUM_CLASSES = 10
# Parameters that can be tuned during different simulations
num_clients = 5
num_selected = 5 # init 20
num_rounds = 3150  # num of communication rounds

epochs = 1			# num of epochs in local client training (An epoch means you go through the entire dataset in that client)
batch_size = 32  # batch size already set in train dataloaderin dataSplit.py
num_batch = 1  # change here choose 100 - 300 default:50

dataset = 'cifar10'
# Meta data setting
meta_sample_number = 1000
meta_bs = 100

# FL model parameters
lr = 0.1  # learning rate
decay_factor = 0.996

# Meta model parameters
meta_net_hidden_size = 100
meta_net_num_layers = 1
meta_lr = 1e-5
meta_weight_decay = 0.


client_weights = np.ones(num_clients)/num_clients
train_dataloaders, test_dataloader, meta_dataloader = get_data_loaders_new(num_clients, batch_size, meta_bs, meta_sample_number)

# global_model = ToyCifarNet().to(device)
# client_models = [ToyCifarNet(init_weights=False).to(device) for _ in range(num_selected)]
global_model = ResNet32(dataset == 'cifar10' and 10 or 100).to(device=device)
client_models = [ResNet32(dataset == 'cifar10' and 10 or 100).to(device=device) for _ in range(num_selected)]

for model in client_models:
    model.load_state_dict(global_model.state_dict())

opt_lst = [optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4) for model in client_models]
criterion = nn.CrossEntropyLoss().to(device)

losses_test = []
acc_test = []

# 初始化元学习网络
meta_net = MLP(hidden_size=meta_net_hidden_size, num_layers=meta_net_num_layers).to(device)
meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=meta_lr, weight_decay=meta_weight_decay)

meta_dataloader_iter = iter(meta_dataloader)

for r in range(num_rounds):
    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print(lr * (decay_factor ** r))
    client_losses = []
    for i in range(num_selected):
        loss = client_update(client_models[i], opt_lst[i], train_dataloaders[i], epochs, num_batch=batch_size)
        client_losses.append(loss)
    print(client_losses)
    client_losses_tensor = torch.tensor(client_losses).view(-1, 1).to(device)
    pseudo_weight = meta_net(client_losses_tensor.data)
    #weights = pseudo_weight / torch.sum(pseudo_weight)
    #print(weights)
    g = make_dot(pseudo_weight)
    g.render(filename=str('./myNetModel'), view=False, format='pdf')
    server_aggregate(global_model, client_models, pseudo_weight)

    # 在元数据集上计算元学习网络的损失
    try:
        meta_inputs, meta_labels = next(meta_dataloader_iter)
    except StopIteration:
        meta_dataloader_iter = iter(meta_dataloader)
        meta_inputs, meta_labels = next(meta_dataloader_iter)

    # 反向传播更新元学习网络
    meta_inputs, meta_labels = meta_inputs.to(device), meta_labels.to(device)
    meta_outputs = global_model(meta_inputs)
    meta_loss = criterion(meta_outputs, meta_labels.long())

    print("Meta_loss: ", meta_loss)
    grad = torch.autograd.grad(meta_loss, meta_net.parameters(), create_graph=True, allow_unused=True)

    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()

    test_loss, acc = test(global_model, test_dataloader)
    print('round %dth average train loss %0.3g | test.py loss %0.3g | test.py acc: %0.3f' % (r, loss / num_selected, test_loss, acc))
    losses_test.append(test_loss)
    acc_test.append(acc)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

    name = "../meta_mat/log_soft_weight_lr" + str(lr) + "_decay" + str(decay_factor) + "_C" + str(
        num_clients) + "_S" + str(num_selected) + "_Nbatch" + str(num_batch) + "_DataSize" + str(
        meta_sample_number) + ".mat"

    sio.savemat(name, {'acc_test': acc_test})

    