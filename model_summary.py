# model_summary.py

from torchinfo import summary
from model import BaselineCNN
import torch


def main():
    model = BaselineCNN(in_channels=3, num_classes=10)

    print("\nModel Architecture:\n")
    print(model)

    print("\nDetailed Summary:\n")

    summary(
        model,
        input_size=(1, 3, 32, 32),  # (batch_size, channels, height, width)
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
            "trainable"
        ],
        depth=5,
        verbose=1
    )


if __name__ == "__main__":
    main()