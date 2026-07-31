import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    return trainset, testset

def stratified_sampling(dataset, num_samples_per_class):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    sampled_indices = []
    for label, indices in class_indices.items():
        sampled_indices.extend(np.random.choice(indices, num_samples_per_class, replace=False))
    
    remaining_indices = [i for i in range(len(dataset)) if i not in sampled_indices]
    return sampled_indices, remaining_indices

def split_dataset(dataset, num_clients, metadata_size=1000):
    num_classes = 10
    num_samples_per_class = metadata_size // num_classes
    
    metadata_indices, remaining_indices = stratified_sampling(dataset, num_samples_per_class)
    metadata = Subset(dataset, metadata_indices)
    
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


def get_data_loaders_new(num_clients=100, metadata_size=1000, batch_size=64):
    set_random_seed(42)
    trainset, testset = load_cifar10()
    metadata, clients_train_data = split_dataset(trainset, num_clients, metadata_size)
    
    print("Metadata class distribution:", count_class_samples(metadata))
    
    client_dataloaders = []
    noise_rates = []
    
    for idx, client_data in enumerate(clients_train_data):
        noise_rate = np.random.uniform(0, 1)
        introduce_label_noise(client_data.dataset, client_data.indices, noise_rate)
        data_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
        client_dataloaders.append(data_loader)
        noise_rates.append(noise_rate)
        print(f"Client {idx+1} noise rate: {noise_rate}")
    
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    meta_dataloader = DataLoader(metadata, batch_size=batch_size, shuffle=True)

    return client_dataloaders, test_dataloader, meta_dataloader
