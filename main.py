from src.data_loader import prepare_dataloaders, get_class_names


def main():
    train_loader, val_loader, test_loader = prepare_dataloaders()
    class_names = get_class_names()

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


if __name__ == "__main__":
    main()