import torch
from data.datamodules.cifar_datamodule import CIFAR10DataModule

# Initialize the DataModule
root_dir = "data/cifar10"
batch_size = 64
num_workers = 0

# Create the DataModule instance
data_module = CIFAR10DataModule(root_dir=root_dir, batch_size=batch_size, num_workers=num_workers)

if __name__ == "__main__":
    # Setup the data module (prepares datasets)
    data_module.setup()

    # Test train dataloader
    print("Testing train dataloader...")
    train_loader = data_module.train_dataloader()
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Train Batch {batch_idx + 1}:")
        print(f"Images shape: {images.shape}")  # Should be [batch_size, 3, 32, 32]
        print(f"Labels: {labels}")
        break  # Test one batch only

    # Test validation dataloader
    print("\nTesting validation dataloader...")
    val_loader = data_module.val_dataloader()
    for batch_idx, (images, labels) in enumerate(val_loader):
        print(f"Validation Batch {batch_idx + 1}:")
        print(f"Images shape: {images.shape}")
        print(f"Labels: {labels}")
        break  # Test one batch only

    # Test test dataloader
    print("\nTesting test dataloader...")
    test_loader = data_module.test_dataloader()
    for batch_idx, (images, labels) in enumerate(test_loader):
        print(f"Test Batch {batch_idx + 1}:")
        print(f"Images shape: {images.shape}")
        print(f"Labels: {labels}")
        break  # Test one batch only
