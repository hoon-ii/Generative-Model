from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms

def load_MNIST():
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(),]
    )
    
    train_dataset = MNIST(
        root='data',
        train=True,
        download=True,
        transform=mnist_transform
    )
    
    test_dataset = MNIST(
        root='data',
        train=False,
        download=True,
        transform=mnist_transform
    )
    
    return train_dataset, test_dataset

def load_CIFAR10():
    cifar10_transform = transforms.Compose(
        [transforms.ToTensor(),]
    )
    
    train_dataset = CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=cifar10_transform
    )
    
    test_dataset = CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=cifar10_transform
    )
    
    return train_dataset, test_dataset