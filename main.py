from src.data_loader import prepare_dataloaders, get_class_names
from src.model import CNNModel
from src.data_loader import prepare_dataloaders
from src.train import train_model
import torch


def main():
    train_loader, val_loader, test_loader = prepare_dataloaders()

    model = train_model(train_loader, val_loader, epochs=5)
    print("\nModel created:")
    print(model)



if __name__ == "__main__":
    main()
