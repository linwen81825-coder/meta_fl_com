import os
from tqdm import tqdm

import numpy as np

import torch

import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Compose

seed = 23333


def get_cifar10():
    '''Return CIFAR10 train/test data and labels as numpy arrays'''
    data_train = datasets.CIFAR10('/root/data', train=True, download=True)
    data_test = datasets.CIFAR10('/root/data', train=False, download=True)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test

def get_mnist():
    
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
    data_train = datasets.FashionMNIST('/root/data', train=True, transform=transform, download=True)
    data_test = datasets.FashionMNIST('/root/data', train=False, transform=transform, download=True)
    
    x_train, y_train = np.array(data_train.data.reshape(data_train.data.shape[0], 1, 28, 28)), np.array(data_train.targets)
    print(x_train.sum())


    x_test, y_test = np.array(data_test.data.reshape(data_test.data.shape[0], 1, 28, 28)), np.array(data_test.targets)
    
    return x_train, y_train, x_test, y_test

def get_svhn():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
    data_train = datasets.SVHN('/root/data', 'train', download=True)
    data_test = datasets.SVHN('/root/data', 'test', download=True)
    data_extra = datasets.SVHN('/root/data', 'extra', download=True)
    
    data_train.data = np.vstack([data_train.data, data_extra.data])
    data_train.labels = np.hstack([data_train.labels, data_extra.labels])
    
    num_train = [0 for i in range(10)]
    for i in range(10):
        for idx, label in enumerate(data_train.labels):
            if label == i:
                num_train[i] += 1

    num_min = min(num_train)
    
    x_train = []
    y_train = []
    for i in range(10):
        for data, label in zip(data_train.data, data_train.labels):

            if label == i and len(y_train) < num_min * (i + 1):
                x_train.append(data)
                y_train.append(label)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    
    num_test = [0 for i in range(10)]
    for i in range(10):
        for idx, label in enumerate(data_test.labels):
            if label == i:
                num_test[i] += 1
    num_min = min(num_test)
    
    x_test = []
    y_test = []
    for i in range(10):
        for data, label in zip(data_test.data, data_test.labels):
            if label == i and len(y_test) < num_min * (i + 1):
                x_test.append(data)
                y_test.append(label)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    
    return x_train, y_train, x_test, y_test
    
    


def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))

def generate_ratio_per_object(ratio=0.5, num_objects=10, fixed_seed_change=23):
    np.random.seed(seed)
    if ratio == 0:
        raise AssertionError

    ratio_per_object = [0 for i in range(num_objects)]

    if ratio == 1.0:
        ratio_per_object = [1 / num_objects for i in range(num_objects)]
    else:

        min_left = ratio / (ratio + num_objects - 1)
        max_right = 1 / (1 + (num_objects - 1) * ratio)
        num_min = np.random.uniform(min_left, 1 / num_objects)
        num_max = num_min / ratio

        while num_max < 1 / num_objects or num_max > max_right:
            num_min = np.random.uniform(min_left, 1 / num_objects)
            num_max = num_min / ratio

        # print(num_min, num_max)
        rng = np.random.default_rng(fixed_seed_change)
        class_idx = rng.permutation(num_objects)
        class_min = class_idx[0]
        class_max = class_idx[1]

        # print(class_min, class_max)

        ratio_per_object[class_min] = num_min
        ratio_per_object[class_max] = num_max

        # print(class_min, num_min, class_max, num_max)

        res = 1 - num_min - num_max
        for i in range(2, num_objects - 1):
            idx = class_idx[i]

            num = np.random.uniform(num_min, num_max)
            res = res - num
            while res < (num_objects - i - 1) * num_min or res > (num_objects - i - 1) * num_max:
                res = res + num
                num = np.random.uniform(num_min, num_max)
                # if res < (num_objects - i - 1) * num_min:
                #     num = np.random.uniform(num_min, num)
                # else:
                #     num = np.random.uniform(num, num_max)
                res = res - num

            ratio_per_object[idx] = num

        ratio_per_object[class_idx[num_objects - 1]] = res

        # print(ratio_per_object)

    return ratio_per_object

def generate_data_by_global_ratio(y_train, global_ratio_per_class, num_classes=10):
    np.random.seed(seed)

    cls2idxd = {}
    num_data_class = [0 for i in range(num_classes)]

    num_data = len(y_train)

    for i in range(num_classes - 1):
        num_data_class[i] = int(global_ratio_per_class[i] * num_data)
    num_data_class[num_classes - 1] = num_data - sum(num_data_class)

    # restrict to less than number of samples per class
    max_num_data_class = max(num_data_class)
    for i in range(num_classes):
        num_data_class[i] = int(num_data_class[i] * num_data / num_classes / max_num_data_class)
    # maybe far from num_data, so uniform add some except min and max
    min_num_data_class_idx = np.argmin(num_data_class)
    max_num_data_class_idx = np.argmax(num_data_class)
    # print(num_data_class)
    for i in range(num_classes):
        if i not in [min_num_data_class_idx, max_num_data_class_idx]:
            if int(num_data / num_classes - num_data_class[i]) == 0:
                num_data_class[i] += 0
            else:
                num_data_class[i] += np.random.randint(0, int(num_data / num_classes - num_data_class[i]))

    for c in range(num_classes):
        cls2idxd[c] = []
        for idx, target in enumerate(y_train):
            if target == c and len(cls2idxd[c]) < num_data_class[c]:
                cls2idxd[c].append(idx)
    print(num_data_class)

    return num_data_class, cls2idxd


def generate_local_ratio(num_clients, fixed_seed_change_lst, num_classes=10, low=.2, high=.8):
    np.random.seed(seed)
    ratio_client_lst = []
    for client in range(num_clients):
        ratio_local = np.random.uniform(low, high)
        ratio_client_lst.append(generate_ratio_per_object(ratio_local, num_classes, fixed_seed_change_lst[client]))
    return ratio_client_lst

def data_break_into(ratio_along_client, num_data_class_scalar):
    num_clients = len(ratio_along_client)
    num_class_client = [0 for i in range(num_clients)]

    for i in range(num_clients - 1):
        num_class_client[i] = int(ratio_along_client[i] * num_data_class_scalar) + np.random.randint(0, 2)
    num_class_client[num_clients - 1] = num_data_class_scalar - sum(num_class_client)

    if num_class_client[num_clients - 1] < 0:
        for client in range(num_clients):
            if num_class_client[client] >= 1:
                num_class_client[client] = num_class_client[client] - 1
                num_class_client[num_clients - 1] = num_class_client[num_clients - 1] + 1
                if num_class_client[num_clients - 1] == 0:
                    break


    # print(num_class_client)

    return num_class_client

def shuffle_list_data(cls2idxd):
    for cls, idxl in cls2idxd.items():
        inds = list(range(len(idxl)))
        np.random.shuffle(inds)
        cls2idxd[cls] = np.asarray(idxl)[inds]

def client_data_split(x_train, y_train, data_client, cls2idxd, num_clients, num_classes=10):
    data = [[[], []] for i in range(num_clients)]
    class_idx_left = [0 for i in range(num_classes)]
    class_idx_right = [0 for i in range(num_classes)]
    for client in range(num_clients):

        for i in range(num_classes):
            class_idx_right[i] = class_idx_left[i] + data_client[client][i]
            idxa = cls2idxd[i][class_idx_left[i]: class_idx_right[i]]
            data[client][0].append(x_train[idxa])
            data[client][1].append(y_train[idxa])
            class_idx_left[i] = class_idx_right[i]

        data[client][0] = np.vstack(data[client][0])
        data[client][1] = np.hstack(data[client][1])

    return data

def distribute_data(ratio_client_lst, num_data_class, perc_client=90, perc_data=100):
    data_client_class = np.zeros_like(ratio_client_lst, dtype='int64')

    ratio_client_arr = np.asarray(ratio_client_lst) # ratio_client_arr: [#clients, #classes]
    
    num_classes = len(num_data_class)
    num_clients = ratio_client_arr.shape[0]
    
    num_data_class_norm = np.array(num_data_class) / np.array(num_data_class).max()
    
        

    perc_client_arr = perc_client * num_data_class_norm
    num_data_class_idx = num_data_class_norm.argsort()
    if num_data_class_norm[num_data_class_idx[0]] > 0.2:
        for i in range(int(num_classes * num_data_class_norm[num_data_class_idx[0]])):
            perc_client_arr[num_data_class_idx[i]] = 15 * (1 + num_data_class_norm[num_data_class_idx[0]])
    

    for i in range(num_classes):
        ratio_class = ratio_client_arr[:, i]
        thres_class = np.percentile(ratio_class, 100 - perc_client_arr[i])

        client_elite_lst = []
        client_loser_lst = []
        for client in range(num_clients):
            if ratio_class[client] >= thres_class:
                client_elite_lst.append(client)
            else:
                client_loser_lst.append(client)

        num_client_elite = len(client_elite_lst)
        data_client_elite = int(num_data_class[i] * (perc_data / 100))
        num_distr_elite_lst = break_into(data_client_elite, num_client_elite)

        num_client_loser = num_clients - num_client_elite
        data_client_loser = num_data_class[i] - data_client_elite
        num_distr_loser_lst = break_into(data_client_loser, num_client_loser)

        for idx, client in enumerate(client_elite_lst):
            data_client_class[client, i] = num_distr_elite_lst[idx]
        for idx, client in enumerate(client_loser_lst):
            data_client_class[client, i] = num_distr_loser_lst[idx]



    return data_client_class


def break_into(n, m):
    '''
    return m random integers with sum equal to n
    '''
    import random
    random.seed(seed)
    to_ret = [1 for i in range(m)]
    for i in range(n - m):
        ind = random.randint(0, m - 1)
        to_ret[ind] += 1
    return to_ret


def data_split(num_clients, ratio=0.5, seed=23333, dataset='cifar'):


    np.random.seed(seed)
    fixed_seed_change = np.random.randint(23, 333333)
    if dataset == 'cifar':
        print('cifar')
        x_train, y_train, x_test, y_test = get_cifar10()
    elif dataset == 'mnist':
        print('mnist')
        x_train, y_train, x_test, y_test = get_mnist()
    elif dataset == 'svhn':
        print('svhn')
        x_train, y_train, x_test, y_test = get_svhn()
    else:
        raise NotImplemented
    global_ratio_per_class = generate_ratio_per_object(ratio=ratio, num_objects=10, fixed_seed_change=fixed_seed_change)
    
    # global_ratio_per_class = [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]
    #global_ratio_per_class = np.zeros(10)
    #for i in range(10):
    #    global_ratio_per_class[i] = 
        
    print('global ratio per class1: ', global_ratio_per_class)   
    
    num_data_class, cls2idxd = generate_data_by_global_ratio(y_train, global_ratio_per_class, num_classes=10)
    shuffle_list_data(cls2idxd)

    fixed_seed_change_lst = [np.random.randint(23, 333333) for i in range(num_clients)]
    ratio_client_lst = generate_local_ratio(num_clients, fixed_seed_change_lst)

    data_client = distribute_data(ratio_client_lst, num_data_class)
    
    print('data_client', data_client)

    clients_split = client_data_split(x_train, y_train, data_client, cls2idxd, num_clients=num_clients, num_classes=10)
    
    clients_split = np.array(clients_split, dtype=object)
    
    

    return clients_split




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


def normalize(data_tensor):
    '''re-scale image values to [-1, 1]'''
    # return (data_tensor / 255.) * 2. - 1. 
    return data_tensor * 2 - 1

def get_data_loaders(nclients, batch_size, real_wd=True, ratio=1, weighted=False, dataset='cifar', seed=23333, classes_pc=10, verbose=True):
    
    if dataset == 'cifar':
        x_train, y_train, x_test, y_test = get_cifar10()
        transforms_train, transforms_eval = get_default_data_transforms(verbose=False)
        print('nononono')
    elif dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist()
        print(x_train.sum())
        
        print('hhhhhh')
    elif dataset == 'svhn':
        x_train, y_train, x_test, y_test = get_svhn()
        print(x_train)
    else:
        raise NotImplemented

    if verbose:
        print_image_data_stats(x_train, y_train, x_test, y_test)

    

    
    clients_split = data_split(nclients, ratio=ratio, seed=seed, dataset=dataset)
    #clients_split = data_split(nclients, ratio=ratio, seed=seed, dataset=dataset)
    
    
    data_size_per_client = []
    for x, y in clients_split:
        data_size_per_client.append(y.shape[0])

    data_size_per_client = np.asarray(data_size_per_client)

    if weighted == False:
        data_size_per_client = np.ones_like(data_size_per_client) * int(y_train.shape[0] / nclients)

    
    if dataset == 'cifar':
        client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train),
                                                  batch_size=batch_size, shuffle=True) for x, y in clients_split] # Q: so you shuffle again?

        test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100,
                                              shuffle=False)
    elif dataset == 'mnist':
        transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
        ])
            
        client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transform),
                                                  batch_size=batch_size, shuffle=True) for x, y in clients_split] # Q: so you shuffle again?

        test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transform), batch_size=100,
                                              shuffle=False)
    elif dataset == 'svhn':
        
        transform=transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: normalize(x)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Lambda(lambda x: normalize(x))
        ])
            
        client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transform),
                                                  batch_size=batch_size, shuffle=True) for x, y in clients_split] # Q: so you shuffle again?

        test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transform), batch_size=100, shuffle=False)
    else:
        raise NotImplemented

    return client_loaders, test_loader, data_size_per_client

def main():
    loaders = get_data_loaders(100, 128, True, 0.01, True, dataset='mnist')
    print('test')

if __name__ == '__main__':
    main()