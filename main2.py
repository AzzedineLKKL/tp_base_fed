import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
import matplotlib.pyplot as plt
from typing import Tuple, List
from PIL import Image
from config import  CLIENTS,ALPHA_DIRICHLET,BATCH_SIZE
# ==============================================
# Task 1: Generating Distributed Datasets
# ==============================================

def generate_distributed_datasets(k: int, alpha: float, save_dir: str) -> List[np.ndarray]:
    """Generate and save non-IID distributed datasets for federated learning.
    
    Args:
        k: Number of clients
        alpha: Dirichlet distribution parameter (controls heterogeneity)
        save_dir: Directory to save client datasets
        
    Returns:
        List of client distributions for visualization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load FashionMNIST without transforms (we'll apply them when loading)
    full_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=None
    )
    
    targets = np.array(full_dataset.targets)
    num_classes = len(full_dataset.classes)
    
    # Get indices for each class
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
    
    # Generate client distributions using Dirichlet
    client_distributions = []
    for _ in range(k):
        proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
        client_distributions.append(proportions)
    
    # Assign samples to clients
    client_indices = [[] for _ in range(k)]
    for class_idx in range(num_classes):
        class_idx_list = class_indices[class_idx]
        np.random.shuffle(class_idx_list)
        
        proportions = np.array([dist[class_idx] for dist in client_distributions])
        proportions = proportions / proportions.sum()  # Normalize
        counts = (proportions * len(class_idx_list)).astype(int)
        counts[-1] += len(class_idx_list) - counts.sum()  # Fix rounding errors
        
        split_indices = np.split(class_idx_list, np.cumsum(counts)[:-1])
        for client_id in range(k):
            client_indices[client_id].extend(split_indices[client_id])
    
    # Save each client's data
    for client_id in range(k):
        client_subset = Subset(full_dataset, client_indices[client_id])
        
        client_data = {
            'data': [x[0] for x in client_subset],  # Save images
            'targets': [x[1] for x in client_subset]  # Save labels
        }
        
        torch.save(client_data, os.path.join(save_dir, f'client_{client_id}.pt'))
    
    return client_distributions

def visualize_client_distributions(client_distributions: List[np.ndarray], num_classes: int):
    """Visualize class distributions across clients."""
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(client_distributions))
    
    for class_idx in range(num_classes):
        proportions = [dist[class_idx] for dist in client_distributions]
        plt.bar(range(len(client_distributions)), proportions, bottom=bottom, label=f'Class {class_idx}')
        bottom += proportions
    
    plt.title('Class Distribution Across Clients')
    plt.xlabel('Client ID')
    plt.ylabel('Proportion of Samples')
    plt.xticks(range(len(client_distributions)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ==============================================
# Task 2: Loading a Client's Dataset
# ==============================================

def load_client_data(cid: int, data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Load a client's dataset and create train/val DataLoaders.
    
    Args:
        cid: Client ID
        data_dir: Directory containing client data
        batch_size: Batch size for DataLoaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load client data with weights_only=False to handle PIL.Image
    client_path = os.path.join(data_dir, f'client_{cid}.pt')
    client_data = torch.load(client_path, weights_only=False)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Apply transforms and create tensors
    images = torch.stack([transform(img) for img in client_data['data']])
    labels = torch.tensor(client_data['targets'])
    
    # Create dataset and split
    dataset = TensorDataset(images, labels)
    val_size = int(len(dataset) * 0.2)  # 20% for validation
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def inspect_client_data(cid: int, train_loader: DataLoader, val_loader: DataLoader):
    """Display information about loaded client data."""
    print(f"\nClient {cid} Dataset Summary:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Show sample images
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(10, 3))
    for i in range(min(5, len(images))):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.suptitle(f"Client {cid} Training Samples")
    plt.show()

# ==============================================
# Main Execution
# ==============================================

def main():
    # Configuration
    NUM_CLIENTS = CLIENTS
    ALPHA = ALPHA_DIRICHLET # Controls data heterogeneity
    SAVE_DIR = './client_datasets'
    
    
    # Task 1: Generate and visualize data
    print("Generating distributed datasets...")
    distributions = generate_distributed_datasets(
        k=NUM_CLIENTS,
        alpha=ALPHA,
        save_dir=SAVE_DIR
    )
    visualize_client_distributions(distributions, num_classes=10)
    
    # Task 2: Load and inspect client data
    print("\nLoading and inspecting client data...")
    for client_id in range(min(3, NUM_CLIENTS)):  # Check first 3 clients
        train_loader, val_loader = load_client_data(
            cid=client_id,
            data_dir=SAVE_DIR,
            batch_size=BATCH_SIZE
        )
        inspect_client_data(client_id, train_loader, val_loader)

if __name__ == "__main__":
    main()