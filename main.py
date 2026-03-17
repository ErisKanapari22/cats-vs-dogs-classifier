from src.data_loader import prepare_dataloaders, get_class_names
from src.model import CNNModel
import torch


def main():
    train_loader, val_loader, test_loader = prepare_dataloaders()
    class_names = get_class_names()
    model = CNNModel()

    print("Data loaders created successfully.\n")

    print(f"Classes: {class_names}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print("\nFirst batch info:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:10].tolist()}")

    print("\nModel created successfully.\n")
    print(model)

    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)

    print("\nTest forward pass:")
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
