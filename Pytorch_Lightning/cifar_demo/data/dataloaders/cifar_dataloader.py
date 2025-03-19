import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from cifar_demo.data.datasets.cifar_dataset import CIFAR10Dataset  # Import your dataset class

class CIFAR10DataLoader:
    def __init__(self, root_dir, batch_size=32, transform=None, num_workers=0):
        # If no transform is passed, we can set a default one
        self.transform = transform or transforms.Compose([
            transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize RGB
        ])
        
        # Create the dataset
        self.dataset = CIFAR10Dataset(root_dir=root_dir, transform=self.transform)
        
        # Create DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_dataloader(self):
        return self.dataloader