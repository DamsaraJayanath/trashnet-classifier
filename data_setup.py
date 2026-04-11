import os
import shutil
import random
from pathlib import Path

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def split_data(dataset_dir: str = "data/dataset-resized",
               output_dir: str = "data/TrashNet",
               split_ratio: float = 0.8,
               seed: int = 42):

    random.seed(seed)
    dataset_path = Path(dataset_dir)
    parent_path = Path(output_dir)

    # Clean previous split to avoid duplicates
    if parent_path.exists():
        print("Cleaning old split data...")
        shutil.rmtree(parent_path)

    train_path = parent_path / "train"
    test_path = parent_path / "test"

    print(f"Splitting dataset ({split_ratio:.0%} train / {1-split_ratio:.0%} test)...")

    for class_dir in sorted(dataset_path.iterdir()):
        if class_dir.is_dir():
            # Safer image filtering
            images = list(class_dir.glob("*.jpg")) + \
                     list(class_dir.glob("*.jpeg")) + \
                     list(class_dir.glob("*.png"))

            random.shuffle(images)

            split_size = int(len(images) * split_ratio)
            train_images = images[:split_size]
            test_images = images[split_size:]

            # IMPORTANT: Create class directories
            (train_path / class_dir.name).mkdir(parents=True, exist_ok=True)
            (test_path / class_dir.name).mkdir(parents=True, exist_ok=True)

            # Copy images
            for img in train_images:
                shutil.copy(img, train_path / class_dir.name / img.name)

            for img in test_images:
                shutil.copy(img, test_path / class_dir.name / img.name)

            print(f"  {class_dir.name:12} → Train: {len(train_images):3d} | Test: {len(test_images):3d}")

    print(" Dataset splitting completed successfully!\n")
    return train_path, test_path


def create_dataloaders(train_dir: str,
                       test_dir: str,
                       batch_size: int = 32,
                       img_size: int = 224):

    #  num_workers
    num_workers = os.cpu_count() if os.cpu_count() is not None else 2

    # Create Transformers (using Augmentation and Normalization)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

    class_names = train_data.classes

    # DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"   Dataloaders created successfully!")
    print(f"   Classes      : {class_names}")
    print(f"   Train images : {len(train_data)}")
    print(f"   Test images  : {len(test_data)}")
    print(f"   Batch size   : {batch_size}")
    print(f"   Num workers  : {num_workers}\n")

    return train_dataloader, test_dataloader, class_names
