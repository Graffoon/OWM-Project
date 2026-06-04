import torch                                          # Core PyTorch library
import torchvision                                    # Provides standard vision datasets like MNIST
import torchvision.transforms as transforms           # Image preprocessing utilities
from torch.utils.data import DataLoader, Subset       # DataLoader wraps a dataset into iterable batches; Subset filters it

def get_split_mnist(task_labels, batch_size=32, train=True):
    """
    Downloads MNIST, flattens the images, and filters them 
    based on the provided task_labels list.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                                  # Converts the PIL image (0-255) to a float tensor (0.0-1.0)
        transforms.Normalize((0.1307,), (0.3081,)),             # Standardises to zero-mean/unit-variance using MNIST's global stats
        transforms.Lambda(lambda x: torch.flatten(x))          # Collapses the (1, 28, 28) tensor into a flat (784,) vector
    ])

    full_dataset = torchvision.datasets.MNIST(
        root='./data',       # Folder where downloaded files are cached
        train=train,         # True = 60,000 training samples; False = 10,000 test samples
        download=True,       # Fetches from the internet if not already on disk
        transform=transform  # Attaches our preprocessing pipeline to every sample
    )

    # Walk through every label and keep only indices whose class is in task_labels
    indices = [i for i, label in enumerate(full_dataset.targets) if label in task_labels]
    task_dataset = Subset(full_dataset, indices)  # Wraps the full dataset so only filtered indices are visible

    # shuffle=True during training randomises order each epoch; False keeps evaluation consistent
    dataloader = DataLoader(task_dataset, batch_size=batch_size, shuffle=train)
    
    return dataloader  # Caller iterates over this to get (inputs, labels) mini-batches
