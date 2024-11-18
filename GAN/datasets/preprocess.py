#%%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloader(batch_size=128, image_size=28):
    """
    Creates DataLoader for MNIST dataset.

    Parameters:
    - batch_size (int): Batch size for training and testing.
    - image_size (int): Size to which each MNIST image will be resized.

    Returns:
    - train_loader (DataLoader): DataLoader for training dataset.
    - test_loader (DataLoader): DataLoader for testing dataset.
    """

    # Define transformation for MNIST images
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and create train and test datasets
    train_dataset = datasets.MNIST(
        root="../mnist_data",
        train=True,
        transform=transform,
        download=True
    )
    test_dataset = datasets.MNIST(
        root="../mnist_data",
        train=False,
        transform=transform,
        download=True
    )

    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    return train_dataset, test_dataset,train_loader, test_loader

#%%
if __name__=='__main__':
    _ = get_mnist_dataloader()