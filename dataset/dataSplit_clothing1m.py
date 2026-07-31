import os
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from collections import defaultdict, Counter
import json


def save_indices(file_path, indices):
    """Save indices to a file."""
    # Convert indices to a list of Python integers
    indices = [int(idx) for idx in indices]

    with open(file_path, 'w') as f:
        json.dump(indices, f)


def load_indices(file_path):
    """Load indices from a file."""
    with open(file_path, 'r') as f:
        indices = json.load(f)
    return indices


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class Clothing1MDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Read the file paths and labels from the annotation file
        with open(annotations_file, 'r') as f:
            for line in f:
                path, label = line.split()
                self.data.append(path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

#
# def load_clothing1m():
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),  # Resize for uniformity
#         transforms.RandomCrop((224, 224)),  # Random crop to match typical input sizes
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     trainset = Clothing1MDataset(
#         root_dir='/home/xm/code/data/clothing1m/raw/Clothing1M/clothing1M',
#         # annotations_file='/home/xm/code/data/clothing1m/raw/Clothing1M/clothing1M/annotations/selected_data_5000_per_class.txt',
#         annotations_file='/home/xm/code/data/clothing1m/raw/Clothing1M/clothing1M/annotations/noisy_label_kv.txt',
#         transform=transform
#     )
#
#     # Assuming a test set is available
#     testset = Clothing1MDataset(
#         root_dir='/home/xm/code/data/clothing1m/raw/Clothing1M/clothing1M',
#         # annotations_file='/home/xm/code/data/clothing1m/raw/Clothing1M/clothing1M/annotations/clean_label_kv.txt',
#         annotations_file='/home/xm/code/data/clothing1m/raw/Clothing1M/clothing1M/annotations/selected_data_500_per_class_clean.txt',
#         transform=transform
#     )
#
#     return trainset, testset


def load_clothing1m():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整缩放尺寸为 128x128
        transforms.RandomCrop((56, 56)),  # 调整随机裁剪尺寸为 112x112
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = Clothing1MDataset(
        root_dir='/home/xm/code/data/clothing1m/raw/Clothing1M/clothing1M',
        annotations_file='/home/xm/code/data/clothing1m/raw/Clothing1M/clothing1M/annotations/noisy_label_kv.txt',
        transform=transform
    )

    # Assuming a test set is available
    testset = Clothing1MDataset(
        root_dir='/home/xm/code/data/clothing1m/raw/Clothing1M/clothing1M',
        annotations_file='/home/xm/code/data/clothing1m/raw/Clothing1M/clothing1M/annotations/selected_data_500_per_class_clean.txt',
        transform=transform
    )
    return trainset, testset
# def stratified_sampling(dataset, num_samples_per_class):
#     class_indices = defaultdict(list)
#     i = 0
#     for idx, (_, label) in enumerate(dataset):
#         print(i)
#         i += 1
#         class_indices[label].append(idx)
#
#     sampled_indices = []
#     for label, indices in class_indices.items():
#         sampled_indices.extend(np.random.choice(indices, num_samples_per_class, replace=False))
#
#     remaining_indices = [i for i in range(len(dataset)) if i not in sampled_indices]
#     return sampled_indices, remaining_indices

def stratified_sampling(dataset, num_samples_per_class):
    class_indices = defaultdict(list)

    # Directly use dataset.labels to avoid unnecessary image processing
    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)

    sampled_indices = []
    for label, indices in class_indices.items():
        sampled_indices.extend(np.random.choice(indices, num_samples_per_class, replace=False))

    remaining_indices = [i for i in range(len(dataset)) if i not in sampled_indices]
    return sampled_indices, remaining_indices


# def split_dataset(dataset, num_clients, metadata_size=1400, num_classes=14, use_dirichlet=False, dirichlet_alpha=0.5):
#     num_samples_per_class = metadata_size // num_classes
#
#     # Perform stratified sampling on the dataset
#     metadata_indices, remaining_indices = stratified_sampling(dataset, num_samples_per_class)
#
#     # Create a metadata subset from the dataset using sampled indices
#     metadata = Subset(dataset, metadata_indices)
#
#     # Divide remaining indices among clients
#     num_items_per_client = len(remaining_indices) // num_clients
#     clients_data = []
#
#     for i in range(num_clients):
#         start_idx = i * num_items_per_client
#         end_idx = start_idx + num_items_per_client if i != num_clients - 1 else len(remaining_indices)
#         subset = Subset(dataset, remaining_indices[start_idx:end_idx])
#         clients_data.append(subset)
#
#     return metadata, clients_data
def split_dataset(dataset, num_clients, metadata_size=1400, num_classes=14, use_dirichlet=True, dirichlet_alpha=0.5):
    num_samples_per_class = metadata_size // num_classes

    # Perform stratified sampling on the dataset for metadata
    metadata_indices, remaining_indices = stratified_sampling(dataset, num_samples_per_class)

    # Create a metadata subset from the dataset using sampled indices
    metadata = Subset(dataset, metadata_indices)

    # Initialize list for client data
    clients_data = [[] for _ in range(num_clients)]

    if use_dirichlet:
        # Calculate the proportion for each client using the Dirichlet distribution
        proportions = np.random.dirichlet([dirichlet_alpha] * num_clients, num_classes)

        class_indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            class_indices[label].append(idx)

        for label, indices in class_indices.items():
            np.random.shuffle(indices)
            # Calculate number of samples for each client
            samples_per_client = (proportions[label] * len(indices)).astype(int)

            # Fix any rounding issues by distributing leftover samples
            for i in range(len(indices) - samples_per_client.sum()):
                samples_per_client[i] += 1

            current_index = 0
            for client_idx in range(num_clients):
                client_indices = indices[current_index:current_index + samples_per_client[client_idx]]
                clients_data[client_idx].extend(client_indices)
                current_index += samples_per_client[client_idx]

    else:
        # Divide remaining indices among clients equally
        num_items_per_client = len(remaining_indices) // num_clients

        for i in range(num_clients):
            start_idx = i * num_items_per_client
            end_idx = start_idx + num_items_per_client if i != num_clients - 1 else len(remaining_indices)
            clients_data[i] = remaining_indices[start_idx:end_idx]

    # Convert lists to Subset objects
    clients_data = [Subset(dataset, client_indices) for client_indices in clients_data]

    return metadata, clients_data




def split_dataset_meta(dataset, metadata_size=1400, num_classes=14):
    num_samples_per_class = metadata_size // num_classes

    # Perform stratified sampling on the dataset
    metadata_indices, remaining_indices = stratified_sampling(dataset, num_samples_per_class)

    # Create a metadata subset from the dataset using sampled indices
    metadata = Subset(dataset, metadata_indices)

    return metadata


def get_data_loaders_clothing1m(num_clients, fl_batch_size, meta_batch_size, metadata_size, use_dirichlet=True, dirichlet_alpha=0.5):
    set_random_seed(50)
    trainset, testset = load_clothing1m()


    # Extract metadata from the test set
    metadata = split_dataset_meta(testset, metadata_size=metadata_size, num_classes=14)

    # Split trainset among clients
    _, clients_train_data = split_dataset(trainset, num_clients, metadata_size=0, num_classes=14, use_dirichlet=True, dirichlet_alpha=0.5)

    client_dataloaders = []
    for idx, client_data in enumerate(clients_train_data):
        data_loader = DataLoader(client_data, batch_size=fl_batch_size, shuffle=True)
        client_dataloaders.append(data_loader)
        print(f"Client {idx + 1} has {len(client_data)} samples.")

    test_dataloader = DataLoader(testset, batch_size=fl_batch_size, shuffle=False)
    meta_dataloader = DataLoader(metadata, batch_size=meta_batch_size, shuffle=False)

    return client_dataloaders, test_dataloader, meta_dataloader

# def get_data_loaders_clothing1m(num_clients, fl_batch_size, meta_batch_size, metadata_size):
#     set_random_seed(50)
#     trainset, testset = load_clothing1m()
#
#     # Define paths to save/load the indices
#     metadata_indices_path = 'metadata_indices.json'
#     client_indices_path = 'client_indices.json'
#
#     # Check if indices are already saved
#     if os.path.exists(metadata_indices_path) and os.path.exists(client_indices_path):
#         # Load saved indices
#         metadata_indices = load_indices(metadata_indices_path)
#         client_indices_list = load_indices(client_indices_path)
#     else:
#         # Split datasets as usual
#         metadata_indices, _ = stratified_sampling(testset, metadata_size // 14)
#         _, remaining_indices = stratified_sampling(trainset, 0)  # Just to get remaining_indices
#         metadata = Subset(testset, metadata_indices)
#
#         num_items_per_client = len(remaining_indices) // num_clients
#         client_indices_list = [remaining_indices[i * num_items_per_client:(i + 1) * num_items_per_client] for i in
#                                range(num_clients)]
#
#         # Save indices for future runs
#         save_indices(metadata_indices_path, metadata_indices)
#         save_indices(client_indices_path, client_indices_list)
#
#     # Create datasets using loaded or newly created indices
#     metadata = Subset(testset, metadata_indices)
#     clients_train_data = [Subset(trainset, client_indices) for client_indices in client_indices_list]
#
#     client_dataloaders = []
#     for idx, client_data in enumerate(clients_train_data):
#         data_loader = DataLoader(client_data, batch_size=fl_batch_size, shuffle=True)
#         client_dataloaders.append(data_loader)
#         print(f"Client {idx + 1} has {len(client_data)} samples.")
#
#     test_dataloader = DataLoader(testset, batch_size=fl_batch_size, shuffle=False)
#     meta_dataloader = DataLoader(metadata, batch_size=meta_batch_size, shuffle=False)
#
#     return client_dataloaders, test_dataloader, meta_dataloader


