import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation block for 1D feature maps.

    Input:
        x: [batch_size, channels, length]

    Output:
        out: [batch_size, channels, length]
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden_channels = max(channels // reduction, 4)

        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _ = x.shape

        weights = self.global_pool(x)          # [batch_size, channels, 1]
        weights = weights.view(batch_size, channels)
        weights = self.fc(weights)             # [batch_size, channels]
        weights = weights.view(batch_size, channels, 1)

        out = x * weights

        return out


class SEBasicBlock1D(nn.Module):
    """
    ResNet1D basic block with SE channel attention.

    Structure:
        Conv1d -> BN -> ReLU -> Conv1d -> BN -> SE -> Residual Add -> ReLU
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, reduction=8):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock1D(
            channels=out_channels,
            reduction=reduction,
        )

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class SEResNet1D(nn.Module):
    """
    SE-ResNet1D for CWRU bearing fault diagnosis.

    Input:
        x: [batch_size, 1, 1024]

    Output:
        logits: [batch_size, num_classes]
    """

    def __init__(self, num_classes=10, reduction=8):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.layer1 = nn.Sequential(
            SEBasicBlock1D(32, 32, stride=1, reduction=reduction),
            SEBasicBlock1D(32, 32, stride=1, reduction=reduction),
        )

        self.layer2 = nn.Sequential(
            SEBasicBlock1D(32, 64, stride=2, reduction=reduction),
            SEBasicBlock1D(64, 64, stride=1, reduction=reduction),
        )

        self.layer3 = nn.Sequential(
            SEBasicBlock1D(64, 128, stride=2, reduction=reduction),
            SEBasicBlock1D(128, 128, stride=1, reduction=reduction),
        )

        self.layer4 = nn.Sequential(
            SEBasicBlock1D(128, 128, stride=2, reduction=reduction),
            SEBasicBlock1D(128, 128, stride=1, reduction=reduction),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)          # [batch_size, 32, 512]
        x = self.layer1(x)        # [batch_size, 32, 512]
        x = self.layer2(x)        # [batch_size, 64, 256]
        x = self.layer3(x)        # [batch_size, 128, 128]
        x = self.layer4(x)        # [batch_size, 128, 64]
        x = self.global_pool(x)   # [batch_size, 128, 1]
        x = x.squeeze(-1)         # [batch_size, 128]
        logits = self.classifier(x)

        return logits


if __name__ == "__main__":
    model = SEResNet1D(num_classes=10)

    dummy_input = torch.randn(8, 1, 1024)
    dummy_output = model(dummy_input)

    print(model)
    print("Input shape :", dummy_input.shape)
    print("Output shape:", dummy_output.shape)