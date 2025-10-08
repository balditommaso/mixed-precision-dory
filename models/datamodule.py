import torch
import pytorch_lightning as pl
from typing import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset


# ---------------------------------------------------------------------------- #
#                                   MNIST                                      #
# ---------------------------------------------------------------------------- #
MNIST_mean = [0.1307]
MNIST_std = [0.3081]

class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 1024,
        val_size: float = 0.2,
        num_workers: int = 8,
        seed: int = 20000605
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.seed = seed

        # transforms
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_mean, MNIST_std),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_mean, MNIST_std)
        ])

    def setup(self, stage: str = None):
        torch.manual_seed(self.seed)
        
        # load full training set
        full_train = datasets.MNIST(
            root=self.data_path, train=True, download=True, transform=self.train_transform
        )
        
        val_len = int(len(full_train) * self.val_size)
        train_len = len(full_train) - val_len
        self.train_dataset, self.val_dataset = random_split(full_train, [train_len, val_len])
        
        # test set
        self.test_dataset = datasets.MNIST(
            root=self.data_path, train=False, download=True, transform=self.test_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    @property
    def dataset_mean(self):
        return MNIST_mean

    @property
    def dataset_std(self):
        return MNIST_std
    
    

# ---------------------------------------------------------------------------- #
#                                   CIFAR-10                                   #
# ---------------------------------------------------------------------------- #
CIFAR10_mean = [0.49139968, 0.48215841, 0.44653091]
CIFAR10_std = [0.2023, 0.1994, 0.2010]

class CIFAR10(datasets.CIFAR10):
    def __init__(
         self,
         root: str, 
         download: bool = True,
         train: bool = True, 
         center: bool = True, 
         rescale: bool = True, 
         augment: bool = True
     ) -> None:

        mean = CIFAR10_mean if center else [0., 0., 0.]
        std = CIFAR10_std if rescale else [1., 1., 1.]
        
        transf_list = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        if train and augment:
            transf_list.append(transforms.RandomCrop(32, padding=4))
            transf_list.append(transforms.RandomHorizontalFlip())

        transform = transforms.Compose(transf_list)
        super().__init__(root=root, train=train, transform=transform, download=download)


   
    
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_path: str,
        batch_size: int=1024,
        val_size: float=0.2,
        num_workers: int=8,
        seed: int = 20000605,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.seed = seed
        self.setup(0)
       
        
    @property
    def dataset_mean():
        return CIFAR10_mean
    
    
    @property
    def dataset_std():
        return CIFAR10_std
        
        
    def setup(self, stage: str) -> None:
        torch.manual_seed(self.seed)
        self.train_dataset = CIFAR10(
            root=self.data_path,
            download=True,
            train=True,
            center=True,
            rescale=True,
            augment=True
        )
        self.val_dataset = CIFAR10(
            root=self.data_path,
            download=True,
            train=True,
            center=True,
            rescale=True,
            augment=False
        )
        self.test_dataset = CIFAR10(
            root=self.data_path,
            download=True,
            train=False,
            center=True,
            rescale=True,
            augment=False
        )
        
        # split the dataset
        train_part, val_part = random_split(self.train_dataset, [1 - self.val_size, self.val_size])
        self.train_dataset = Subset(self.train_dataset, train_part.indices)
        self.val_dataset = Subset(self.val_dataset, val_part.indices)
        self.summary()
    
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
    
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        ) 
        
    def summary(self):
        print(f"** CIFAR-10 dataset: **\n" \
              f"train dataset:\t{len(self.train_dataset)}\n" \
              f"val dataset:\t{len(self.val_dataset)}\n" \
              f"test dataset:\t{len(self.test_dataset)}\n")


DATALOADERS = {
    "cifar-10": CIFAR10DataModule,
    "MNIST": MNISTDataModule
}
