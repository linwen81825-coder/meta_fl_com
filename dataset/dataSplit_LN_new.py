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
 
 

def load_cinic10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4789, 0.4723, 0.4305),
            (0.2421, 0.2383, 0.2587)
        )
    ])

    trainset = torchvision.datasets.ImageFolder(
        root='data/CINIC-10/train',
        transform=transform
    )

    testset = torchvision.datasets.ImageFolder(
        root='data/CINIC-10/test',
        transform=transform
    )

    trainset.targets = [
        label for _, label in trainset.samples
    ]

    testset.targets = [
        label for _, label in testset.samples
    ]

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


def apply_long_tail_distribution(dataset, remaining_indices, metadata_size, num_classes, imbalanced_factor):
    """
    使用项目 noisy_long_tail_CIFAR.py 中原有的长尾规则：
        n_c = sample_num / imbalanced_factor ** (c / (C - 1))

    与原项目保持一致，对 imbalanced_num_list 做随机 shuffle，
    因此具体哪个类别成为 head / tail 是随机的。

    imbalanced_factor=None 时不启用长尾。
    """
    if imbalanced_factor is None:
        return remaining_indices

    sample_num = int((len(dataset.targets) - metadata_size) / num_classes)

    imbalanced_num_list = []
    for class_index in range(num_classes):
        imbalanced_num = sample_num / (
            imbalanced_factor ** (class_index / (num_classes - 1))
        )
        imbalanced_num_list.append(int(imbalanced_num))

    # 与 noisy_long_tail_CIFAR.py 原逻辑一致
    np.random.shuffle(imbalanced_num_list)
    print("Long-tail class sample targets:", imbalanced_num_list)

    targets = np.array(dataset.targets)
    remaining_indices = np.array(remaining_indices, dtype=np.int64)

    long_tail_indices = []

    for class_index in range(num_classes):
        class_indices = remaining_indices[
            targets[remaining_indices] == class_index
        ].copy()

        np.random.shuffle(class_indices)

        class_indices = class_indices[
            :imbalanced_num_list[class_index]
        ]

        long_tail_indices.extend(class_indices.tolist())

    np.random.shuffle(long_tail_indices)

    print(
        "Long-tail total client samples before equal split:",
        len(long_tail_indices)
    )

    return long_tail_indices
 
 
def split_dataset(
    dataset,
    num_clients,
    metadata_size=1000,
    use_dirichlet=False,
    dirichlet_alpha=0.5,
    imbalanced_factor=None
): 
    if dataset == 'cifar100': 
        num_classes = 100 
    else: 
        num_classes = 10 
 
    num_samples_per_class = metadata_size // num_classes 
 
    # 获取元数据集和剩余数据集 
    metadata_indices, remaining_indices = stratified_sampling(dataset, num_samples_per_class) 
    metadata = Subset(dataset, metadata_indices) 

    # 只对客户端训练数据构造长尾；metadata 仍保持原来的均衡抽样
    remaining_indices = apply_long_tail_distribution(
        dataset,
        remaining_indices,
        metadata_size,
        num_classes,
        imbalanced_factor
    )
 
    if use_dirichlet: 
        # Dirichlet Non-IID + 每个客户端样本数量严格相同
        print("use_dirichlet")

        num_items_per_client = len(remaining_indices) // num_clients
        total_used_samples = num_items_per_client * num_clients

        remaining_indices = np.array(remaining_indices, dtype=np.int64)
        np.random.shuffle(remaining_indices)

        dropped_samples = len(remaining_indices) - total_used_samples
        remaining_indices = remaining_indices[:total_used_samples]

        remaining_targets = np.array([
            dataset.targets[i] for i in remaining_indices
        ])

        client_indices = [[] for _ in range(num_clients)]

        # 每个客户端最终容量完全相同
        remaining_capacity = np.full(
            num_clients,
            num_items_per_client,
            dtype=np.int64
        )

        # 每个类别分别产生 Dirichlet 客户端偏好
        for class_id in range(num_classes):
            class_positions = np.where(
                remaining_targets == class_id
            )[0]

            class_sample_indices = remaining_indices[
                class_positions
            ].copy()

            np.random.shuffle(class_sample_indices)

            if len(class_sample_indices) == 0:
                continue

            proportions = np.random.dirichlet(
                np.repeat(dirichlet_alpha, num_clients)
            )

            # Dirichlet 决定类别偏好；capacity 保证客户端总样本数一致
            for sample_idx in class_sample_indices:
                available = remaining_capacity > 0

                probs = proportions.copy()
                probs[~available] = 0.0

                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    client_idx = np.random.choice(
                        num_clients,
                        p=probs
                    )
                else:
                    available_clients = np.where(available)[0]
                    client_idx = np.random.choice(available_clients)

                client_indices[client_idx].append(int(sample_idx))
                remaining_capacity[client_idx] -= 1

        client_sizes = [len(indices) for indices in client_indices]

        print("Client sample sizes:", client_sizes)
        print("Samples per client:", num_items_per_client)
        print("Dropped samples for equal client sizes:", dropped_samples)

        assert all(
            size == num_items_per_client
            for size in client_sizes
        ), f"Client sample sizes are not equal: {client_sizes}"

        clients_data = [
            Subset(dataset, indices)
            for indices in client_indices
        ]
 
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
 
 
def get_data_loaders_new(
    num_clients,
    fl_batch_size,
    meta_batch_size,
    metadata_size,
    dataset='cifar10',
    isnoise=True,
    use_dirichlet=False,
    dirichlet_alpha=0.5,
    imbalanced_factor=None
): 
    set_random_seed(649) 
    if dataset == 'cifar10': 
        trainset, testset = load_cifar10() 
    elif dataset == 'cifar100': 
        trainset, testset = load_cifar100() 
    elif dataset == 'cinic10':
        trainset, testset = load_cinic10()
 
    metadata, clients_train_data = split_dataset(
        trainset,
        num_clients,
        metadata_size,
        use_dirichlet,
        dirichlet_alpha,
        imbalanced_factor
    ) 
 
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
