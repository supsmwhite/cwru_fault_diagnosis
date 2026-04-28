import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    1D-CNN baseline model for CWRU bearing fault diagnosis.

    Input:
        x: [batch_size, 1, 1024]

    Output:
        logits: [batch_size, num_classes]
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(output_size=1),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)      # [batch_size, 128, 1]
        x = x.squeeze(-1)         # [batch_size, 128]
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    model = CNN1D(num_classes=10)

    dummy_input = torch.randn(8, 1, 1024)
    dummy_output = model(dummy_input)

    print(model)
    print("Input shape :", dummy_input.shape)
    print("Output shape:", dummy_output.shape)