import random
import numpy as np
import torch

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.config import (
    DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    VAL_SPLIT,
    RANDOM_SEED,
    MEAN,
    STD,
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def load_datasets():
    trainval_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="binary-category",
        transform=get_train_transforms(),
        download=True
    )

    test_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="test",
        target_types="binary-category",
        transform=get_eval_transforms(),
        download=True
    )

    return trainval_dataset, test_dataset


def split_train_validation(trainval_dataset):
    total_size = len(trainval_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, val_dataset = random_split(
        trainval_dataset,
        [train_size, val_size],
        generator=generator
    )

    base_eval_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="binary-category",
        transform=get_eval_transforms(),
        download=False
    )

    val_dataset.dataset = base_eval_dataset

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader


def get_class_names():
    return ["Cat", "Dog"]


def prepare_dataloaders():
    set_seed()

    trainval_dataset, test_dataset = load_datasets()
    train_dataset, val_dataset = split_train_validation(trainval_dataset)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )

    return train_loader, val_loader, test_loader
