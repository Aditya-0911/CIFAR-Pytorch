import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from data import get_dataloaders
from model import BaselineCNN,ResnetTransfer, ResNetCIFAR
from train import train_model,evaluate


def main():

    # ----Device Setup----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "resnet", "resnet_scratch"]
        help="Choose model architecture"
    )

    args = parser.parse_args()

    if args.model == "baseline":
        image_size = 32
    else:
        image_size = 224

    # ----Data Loaders----
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128,num_workers=2, image_size=image_size)

    # ----Model, Loss, Optimizer----
    if args.model == "baseline":
        model = BaselineCNN(in_channels=3, num_classes=10)

    elif args.model == "resnet":
        model = ResnetTransfer(num_classes=10)

    elif args.model == "resnet_scratch":
        model = ResNetCIFAR(num_classes=10)

    model = model.to(device)

    print(next(model.parameters()).device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10,
        gamma=0.1
    ) 


    # ----Train the Model----
    epochs = 20
    patience = 7

    print(f"Training model: {args.model}")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        num_epochs=epochs,
        patience=patience
    )

    # --load best model for testing--

    model.load_state_dict(torch.load("best_model.pth"))


    #----Evaluate on Test Set----
    test_loss, test_acc = evaluate(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device
    )

    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()