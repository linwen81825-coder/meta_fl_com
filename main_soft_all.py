import scipy.io as sio

# modules
from model import ToyCifarNet
# from sampling import uniform_sampling

from sampling import *
from dataset.dataSplit_LN_new import get_data_loaders_new
from model_operation import *

# GPU settings
torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

NUM_CLASSES = 10
# Parameters that can be tuned during different simulations
num_clients = 10
num_selected = 10  # init 20
num_rounds = 3150  # num of communication rounds

epochs = 5			# num of epochs in local client training (An epoch means you go through the entire dataset in that client)
batch_size = 10  # batch size already set in train dataloaderin dataSplit.py
num_batch = 10  # change here choose 100 - 300 default:50
meta_sample_number = 1000
meta_bs = 64

# hyperparameters of deep models
lr = 0.1  # learning rate
decay_factor = 0.996

client_weights = np.ones(num_clients)/num_clients
train_dataloaders, test_dataloader, meta_dataloader = get_data_loaders_new(num_clients, metadata_size=meta_sample_number, batch_size=meta_bs)

global_model = ToyCifarNet().to(device)
client_models = [ToyCifarNet(init_weights=False).to(device) for _ in range(num_selected)]

for model in client_models:
    model.load_state_dict(global_model.state_dict())

opt_lst = [optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4) for model in client_models]
criterion = nn.CrossEntropyLoss()

losses_test = []
acc_test = []

for r in range(num_rounds):
    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print(lr * (decay_factor ** r))
    client_losses = []
    for i in range(num_selected):
        loss = client_update(client_models[i], opt_lst[i], train_dataloaders[i], epochs, num_batch=batch_size)
        client_losses.append(loss)
    print(client_losses)
    server_aggregate(global_model, client_models, client_weights)
    test_loss, acc = test(global_model, test_dataloader)
    print('round %dth average train loss %0.3g | test.py loss %0.3g | test.py acc: %0.3f' % (r, loss / num_selected, test_loss, acc))
    losses_test.append(test_loss)
    acc_test.append(acc)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

    name = "../meta_mat/log_all_lr" + str(lr) + "_decay" + str(decay_factor)  + "_C" + str(
        num_clients) + "_S" + str(num_selected) + "_Nbatch" + str(num_batch) + "_DataSize" + str(meta_sample_number) +  ".mat"

    sio.savemat(name, {'acc_test': acc_test})

