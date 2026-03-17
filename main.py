from src.data_loader import prepare_dataloaders
from src.train import train_model, get_loss_function
from src.evaluate import evaluate_model, plot_metrics


def main():
    train_loader, val_loader, test_loader = prepare_dataloaders()

    model, train_losses, val_losses, train_accs, val_accs = train_model(
        train_loader, val_loader, epochs=5
    )

    loss_fn = get_loss_function()

    evaluate_model(model, test_loader, loss_fn)

    plot_metrics(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    main()