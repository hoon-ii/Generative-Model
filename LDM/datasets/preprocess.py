#%%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CIFAR_Labeler(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = self.classes[target]
        return img, target
    
def get_dataset(dataset='mnist', image_size=28):
    if dataset == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

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

    if dataset =='cifar10':
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean , std )
        ])

        # Download and create train and test datasets
        train_dataset = CIFAR_Labeler(
            root="../cifar_data",
            train=True,
            transform=transform,
            download=True
        )
        test_dataset = CIFAR_Labeler(
            root="../cifar_data",
            train=False,
            transform=transform,
            download=True
        )

    return train_dataset, test_dataset, mean, std


def get_dataloader(train_dataset, test_dataset, dataset='mnist', batch_size=128):
    if dataset == 'mnist':
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
    if dataset =='cifar10':
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

    return train_loader, test_loader


#%%
if __name__=='__main__':
    tr_ds, te_ds, mean, std = get_dataset('mnist',28)
    tr_ld, ts_ld = get_dataloader(tr_ds, 128)