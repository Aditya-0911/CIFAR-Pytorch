import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloaders
from model import BaselineCNN
from train import train_model,evaluate


def main():

    # ----Device Setup----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----Data Loaders----
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128,num_workers=4)

    # ----Model, Loss, Optimizer----
    model = BaselineCNN(in_channels=3, num_classes=10).to(device)
    print(next(model.parameters()).device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10,
        gamma=0.1
    ) 


    # ----Train the Model----
    epochs = 20
    patience = 7

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