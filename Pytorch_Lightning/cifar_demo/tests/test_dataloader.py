from cifar_demo.data.dataloaders.cifar_dataloader import CIFAR10DataLoader

# Initialize the DataLoader
dataloader = CIFAR10DataLoader(root_dir="data/cifar10/train", batch_size=64)

# Test DataLoader
if __name__ == "__main__":
    for batch_idx, (images, labels) in enumerate(dataloader.get_dataloader()):
        print(f"Batch {batch_idx + 1}:")
        print(f"Images shape: {images.shape}")  # Should be [64, 3, 32, 32]
        print(f"Labels: {labels}")
        break  # Test one batch only
