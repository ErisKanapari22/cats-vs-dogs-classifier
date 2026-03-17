import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from src.model import CNNModel
from src.config import DEVICE, MODEL_DIR


def get_model():
    model = CNNModel().to(DEVICE)
    return model


def get_loss_function():
    return nn.BCEWithLogitsLoss()


def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.001)


def calculate_accuracy(outputs, labels):
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()

    correct = (preds == labels.unsqueeze(1)).sum().item()
    return correct


def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader):
        images = images.to(DEVICE)
        labels = labels.float().to(DEVICE)

        outputs = model(images)
        loss = loss_fn(outputs, labels.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += calculate_accuracy(outputs, labels)
        total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def validate(model, loader, loss_fn):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, labels.unsqueeze(1))

            total_loss += loss.item()
            total_correct += calculate_accuracy(outputs, labels)
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train_model(train_loader, val_loader, epochs=10):
    model = get_model()
    loss_fn = get_loss_function()
    optimizer = get_optimizer(model)

    best_val_acc = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer
        )

        val_loss, val_acc = validate(
            model, val_loader, loss_fn
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
            print("✅ Model saved!")

    return model
