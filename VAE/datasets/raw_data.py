from torchvision.datasets import MNIST
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