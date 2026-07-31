import os
from tqdm import tqdm

import numpy as np

import torch

import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Compose



def random_seed():
    return 23333


def get_cifar10():
    '''Return CIFAR10 train/test data and labels as numpy arrays'''
    data_train = datasets.CIFAR10('../../data', train=True, download=True)
    data_test = datasets.CIFAR10('../../data', train=False, download=True)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test

def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))

def clients_rand(train_len, num_clients, rand_partition = True):
    '''
    train_len: size of the train data
    nclients: number of clients
    (Not implemented) weight: user-defined parameter, distrute the number of examples to clients
    rand_partition: modified original random partition version by @threeWater, just set True now.

    Returns: to_ret

    This function creates a random distribution
    for the clients, i.e. number of images each client
    possess.

    '''

    # this part when random = True, contributed by @threeWater
    # there's little issue in original version:
    # 		always output a small number no matter which num_clients chose
    # 		maybe caused by numerial errors
    # [DONE] fixed in this version
    if rand_partition: # True by default
        client_tmp = []
        sum_ = 0
        #### creating random values for each client ####
        for i in range(num_clients - 1):
            tmp = np.random.randint(10, 100) # fucking hardcode
            sum_ += tmp
            client_tmp.append(tmp)

        client_tmp = np.array(client_tmp)
        #### using those random values as weights ####
        # error in this expression
        # clients_dist = ((client_tmp / sum_) * train_len).astype(int)
        # modified a little to
        clients_dist = ((client_tmp / (sum_ + 50)) * train_len).astype(int) # "sum_+50" hardcoded
        num = train_len - clients_dist.sum()
        to_ret = list(clients_dist)
        to_ret.append(num)
        return to_ret


def split_image_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
    '''
    Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
    Input:
      data : [n_data x shape]
      labels : [n_data (x 1)] from 0 to n_labels
      n_clients : number of clients
      classes_per_client : number of classes per client
      shuffle : True/False => True for shuffling the dataset, False otherwise
      verbose : True/False => True for printing some info, False otherwise
    Output:
      clients_split : client data into desired format
    '''
    #### constants ####
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    ### client distribution ####
    data_per_client = clients_rand(len(data), n_clients)
    # data_per_client = [490] * 50 + [510] * 50
    # print(data_per_client)
    data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []

        budget = data_per_client[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

    if verbose:
        print_split(clients_split)

    # print(len(clients_split), len(clients_split[0]), len(clients_split[0][0]), clients_split[0][0].shape, clients_split[0][1])
    clients_split = np.array(clients_split)
    # print(clients_split.shape, clients_split[0][0].shape)

    return clients_split


def add_uniform_noise(y_train, noise_ratio=0.5, num_classes=10):
    y_train_tensor = torch.tensor(y_train, requires_grad=False)
    transition_matrix = torch.eye(num_classes)
    transition_matrix.mul_(1 - noise_ratio * (num_classes / (num_classes - 1)))
    transition_matrix.add_(noise_ratio / (num_classes))


    transition_targets = torch.distributions.Categorical(
        probs=transition_matrix[y_train_tensor, :]).sample()

    y_train_noisy = transition_targets.numpy()

    return y_train_noisy

def generate_noisy_data(y_train, clients_split):

    y_train_noisy = add_uniform_noise(y_train)
    clients_split_noisy = []
    noise_ratio_lst = []
    for data, target in clients_split:
        ratio = np.random.uniform(0, 1)
        noise_ratio_lst.append(ratio)
        target_noisy = add_uniform_noise(target, noise_ratio=ratio)
        clients_split_noisy.append([data, target_noisy])

    print('-------------------------------------------------')
    print('noise_ratio_lst: ', noise_ratio_lst)
    print('-------------------------------------------------')
    return clients_split_noisy, noise_ratio_lst

class CustomImageDataset(Dataset):
  '''
  A custom Dataset class for images
  inputs : numpy array [n_data x shape]
  labels : numpy array [n_data (x 1)]
  '''
  def __init__(self, inputs, labels, transforms=None):
      assert inputs.shape[0] == labels.shape[0]
      self.inputs = torch.Tensor(inputs)
      self.labels = torch.Tensor(labels).long()
      self.transforms = transforms

  def __getitem__(self, index):
      img, label = self.inputs[index], self.labels[index]

      if self.transforms is not None:
        img = self.transforms(img)

      return (img, label)

  def __len__(self):
      return self.inputs.shape[0]

def get_default_data_transforms(train=True, verbose=True):
  transforms_train = {
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
  }
  transforms_eval = {
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  }
  if verbose:
    print("\nData preprocessing: ")
    for transformation in transforms_train['cifar10'].transforms:
      print(' -', transformation)
    print()

  return (transforms_train['cifar10'], transforms_eval['cifar10'])


def get_data_loaders(nclients, batch_size, real_wd=True, weighted=False, classes_pc=10, verbose=True):

    # set seed
    np.random.seed(random_seed())
    torch.manual_seed(random_seed())
    torch.cuda.manual_seed_all(random_seed())

    x_train, y_train, x_test, y_test = get_cifar10()

    if verbose:
        print_image_data_stats(x_train, y_train, x_test, y_test)

    transforms_train, transforms_eval = get_default_data_transforms(verbose=False)

    clients_split = split_image_data(x_train, y_train, n_clients=nclients,
                             classes_per_client=classes_pc, verbose=verbose)

    clients_split_noisy, noise_ratio_lst = generate_noisy_data(y_train, clients_split) # [TODO] remember I return noise_ratio_lst here

    data_size_per_client = np.array([clients_split_noisy[i][0].shape[0] for i in range(nclients)])

    if weighted is False:
        data_size_per_client = np.ones_like(data_size_per_client)

    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train),
                                                  batch_size=batch_size, shuffle=True) for x, y in clients_split_noisy]

    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100,
                                              shuffle=False)

    return client_loaders, test_loader, data_size_per_client

# def main():
#     get_data_loaders(100, 128)


# if __name__ == '__main__':
#     main()