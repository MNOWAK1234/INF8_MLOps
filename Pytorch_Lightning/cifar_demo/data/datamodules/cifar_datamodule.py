import pytorch_lightning as pl
from cifar_demo.data.dataloaders.cifar_dataloader import CIFAR10DataLoader

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, root_dir="cifar_demo/data/cifar10", batch_size=32, num_workers=0, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform  

    def setup(self, stage=None):
        """Set up DataLoaders for training, validation, and testing."""
        self.train_loader = CIFAR10DataLoader(f"{self.root_dir}/train", self.batch_size, self.transform, self.num_workers).get_dataloader()
        self.val_loader = CIFAR10DataLoader(f"{self.root_dir}/val", self.batch_size, self.transform, self.num_workers).get_dataloader()
        self.test_loader = CIFAR10DataLoader(f"{self.root_dir}/test", self.batch_size, self.transform, self.num_workers).get_dataloader()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
