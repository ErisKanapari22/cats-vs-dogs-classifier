import torch
import matplotlib.pyplot as plt

from src.config import DEVICE, OUTPUTS_DIR


def evaluate_model(model, test_loader, loss_fn):

    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, labels.unsqueeze(1))

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            total_loss += loss.item()
            total_correct += (preds == labels.unsqueeze(1)).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples

    print("\n📊 Test Results:")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)


    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(OUTPUTS_DIR / "loss_plot.png")


    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(OUTPUTS_DIR / "accuracy_plot.png")

    print("\n📈 Plots saved in outputs/")

