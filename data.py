import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split

# CIFAR-10 dataset specific transforms 

def get_transforms(image_size=32):
    """
    Returns train and test transforms for CIFAR-10.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, test_transform


def get_dataloaders(batch_size=128, num_workers=2, data_dir="./data", image_size=32):

    train_transform, test_transform = get_transforms(image_size=image_size)

    # Load full training dataset WITHOUT transform first
    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Now assign transforms separately
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = test_transform

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader