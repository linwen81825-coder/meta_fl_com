import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict, Counter


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='/home/lw/Project/data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='/home/lw/Project/data', train=False, download=True, transform=transform)
    return trainset, testset

def load_cifar100():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR100(root='/home/lw/Project/data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='/home/lw/Project/data', train=False, download=True, transform=transform)
    return trainset, testset


def stratified_sampling(dataset, num_samples_per_class):
    # Convert dataset targets to a NumPy array for faster processing
    targets = np.array(dataset.targets)

    # Use a defaultdict to store indices of each class
    class_indices = defaultdict(list)

    # Populate the class_indices dictionary with indices of each class
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    # Initialize list for sampled indices
    sampled_indices = []

    # Sample indices from each class
    for label, indices in class_indices.items():
        sampled_indices.extend(np.random.choice(indices, num_samples_per_class, replace=False))

    # Calculate remaining indices that are not sampled
    remaining_indices = list(set(range(len(dataset))) - set(sampled_indices))

    return sampled_indices, remaining_indices


def split_dataset(dataset, num_clients, metadata_size=1000, use_dirichlet=False, dirichlet_alpha=0.5):
    if dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10

    num_samples_per_class = metadata_size // num_classes

    # 获取元数据集和剩余数据集
    metadata_indices, remaining_indices = stratified_sampling(dataset, num_samples_per_class)
    metadata = Subset(dataset, metadata_indices)

    if use_dirichlet:
        # 通过狄利克雷分布分配剩余数据
        print("use_dirichlet")
        remaining_targets = np.array([dataset.targets[i] for i in remaining_indices])
        class_indices = [np.where(remaining_targets == i)[0] for i in range(num_classes)]

        client_indices = [[] for _ in range(num_clients)]
        for i, indices in enumerate(class_indices):
            proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
            proportions = np.cumsum(proportions)
            proportions = (proportions * len(indices)).astype(int)[:-1]
            split_indices = np.split(indices, proportions)
            for client_idx, split in enumerate(split_indices):
                client_indices[client_idx].extend(split)

        clients_data = [Subset(dataset, [remaining_indices[i] for i in indices]) for indices in client_indices]
    else:
        # 均匀分配剩余数据
        print("uniform")
        num_items_per_client = len(remaining_indices) // num_clients
        clients_data = []
        for i in range(num_clients):
            start_idx = i * num_items_per_client
            end_idx = start_idx + num_items_per_client if i != num_clients - 1 else len(remaining_indices)
            subset = Subset(dataset, remaining_indices[start_idx:end_idx])
            clients_data.append(subset)

    return metadata, clients_data


def introduce_label_noise(dataset, indices, noise_rate):
    """在子数据集中引入标签噪声"""
    targets = np.array(dataset.targets)
    num_noisy_labels = int(noise_rate * len(indices))
    noisy_indices = np.random.choice(indices, num_noisy_labels, replace=False)
    new_labels = np.random.randint(0, 10, num_noisy_labels)  # CIFAR-10有10个类别
    for idx, noisy_idx in enumerate(noisy_indices):
        targets[noisy_idx] = new_labels[idx]
    dataset.targets = targets.tolist()


def count_class_samples(dataset):
    """统计每个类别的样本数量"""
    class_counts = {}
    for data, target in dataset:
        if target not in class_counts:
            class_counts[target] = 0
        class_counts[target] += 1
    return class_counts


def get_dataloader_info(dataloader):
    """
    获取DataLoader中的数据数量和类别分布，并打印结果

    参数:
    dataloader (torch.utils.data.DataLoader): 数据加载器

    返回:
    data_count (int): 数据总量
    label_distribution (dict): 类别分布
    """
    # 获取数据数量
    data_count = len(dataloader.dataset)

    # 获取类别分布
    labels = []
    for _, target in dataloader:
        labels.extend(target.numpy())

    label_distribution = dict(Counter(labels))

    # 打印数据总量和类别分布
    print(f"数据总量: {data_count}")
    print("类别分布:")
    for label, count in label_distribution.items():
        print(f"类别 {label}: {count} 个样本")

    return data_count, label_distribution


def get_data_loaders_new(num_clients, fl_batch_size, meta_batch_size, metadata_size, dataset='cifar10', isnoise=True,
                         use_dirichlet=False, dirichlet_alpha=0.5):
    set_random_seed(50)
    if dataset == 'cifar10':
        trainset, testset = load_cifar10()
    elif dataset == 'cifar100':
        trainset, testset = load_cifar100()

    metadata, clients_train_data = split_dataset(trainset, num_clients, metadata_size, use_dirichlet, dirichlet_alpha)

    print("Metadata class distribution:", count_class_samples(metadata))

    client_dataloaders = []
    noise_rates = []
    if isnoise:
        for idx, client_data in enumerate(clients_train_data):
            noise_rate = np.random.uniform(0, 1)
            introduce_label_noise(client_data.dataset, client_data.indices, noise_rate)
            data_loader = DataLoader(client_data, batch_size=fl_batch_size, shuffle=True)
            client_dataloaders.append(data_loader)
            noise_rates.append(noise_rate)
            print(f"Client {idx + 1} noise rate: {noise_rate}")
    else:
        for idx, client_data in enumerate(clients_train_data):
            data_loader = DataLoader(client_data, batch_size=fl_batch_size, shuffle=True)
            client_dataloaders.append(data_loader)

    test_dataloader = DataLoader(testset, batch_size=fl_batch_size, shuffle=False)
    meta_dataloader = DataLoader(metadata, batch_size=meta_batch_size, shuffle=False)

    return client_dataloaders, test_dataloader, meta_dataloader

