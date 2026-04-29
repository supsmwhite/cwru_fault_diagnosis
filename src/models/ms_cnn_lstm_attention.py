import torch
import torch.nn as nn


def get_group_count(channels, max_groups=8):
    for groups in range(max_groups, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNReLU(nn.Module):
    """
    Conv1d + GroupNorm + ReLU.

    GroupNorm is used instead of BatchNorm to reduce batch-statistics
    instability in few-shot and augmentation-heavy experiments.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.0):
        super().__init__()

        padding = kernel_size // 2
        groups = get_group_count(out_channels)

        layers = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """
    Lightweight residual downsampling block for 1D temporal features.
    """

    def __init__(self, channels, kernel_size, stride=2, dropout=0.0):
        super().__init__()

        self.main = nn.Sequential(
            ConvGNReLU(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
            ),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=get_group_count(channels),
                num_channels=channels,
            ),
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=get_group_count(channels),
                num_channels=channels,
            ),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main(x)
        identity = self.shortcut(x)
        return self.relu(out + identity)


class MultiScaleStem(nn.Module):
    """
    Parallel 1D convolution branches with different receptive fields.

    Small kernels capture short impacts; larger kernels capture wider local
    waveform patterns. The fused sequence is passed to BiLSTM afterward.
    """

    def __init__(self, out_channels_per_branch=32, dropout=0.0):
        super().__init__()

        self.branch3 = ConvGNReLU(
            in_channels=1,
            out_channels=out_channels_per_branch,
            kernel_size=3,
            stride=2,
            dropout=dropout,
        )
        self.branch5 = ConvGNReLU(
            in_channels=1,
            out_channels=out_channels_per_branch,
            kernel_size=5,
            stride=2,
            dropout=dropout,
        )
        self.branch7 = ConvGNReLU(
            in_channels=1,
            out_channels=out_channels_per_branch,
            kernel_size=7,
            stride=2,
            dropout=dropout,
        )

    def forward(self, x):
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        return torch.cat([x3, x5, x7], dim=1)


class TemporalAttention(nn.Module):
    """
    Attention pooling over BiLSTM temporal outputs.
    """

    def __init__(self, hidden_dim, attention_dim=64, dropout=0.0):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, lstm_out):
        scores = self.attention(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        return context, weights


class MSCNNLSTMAttention(nn.Module):
    """
    Multi-scale CNN-BiLSTM-Attention for CWRU bearing fault diagnosis.

    Input:
        x: [batch_size, 1, 1024]

    Output:
        logits: [batch_size, num_classes]
    """

    def __init__(
        self,
        num_classes=10,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        attention_dim=64,
        dropout=0.3,
    ):
        super().__init__()

        self.stem = MultiScaleStem(
            out_channels_per_branch=32,
            dropout=0.0,
        )  # [B, 96, 512]

        self.fusion = ConvGNReLU(
            in_channels=96,
            out_channels=128,
            kernel_size=1,
            stride=1,
            dropout=dropout,
        )  # [B, 128, 512]

        self.down1 = ResidualConvBlock(
            channels=128,
            kernel_size=5,
            stride=2,
            dropout=dropout,
        )  # [B, 128, 256]

        self.down2 = ResidualConvBlock(
            channels=128,
            kernel_size=3,
            stride=2,
            dropout=dropout,
        )  # [B, 128, 128]

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if lstm_num_layers == 1 else dropout,
        )

        lstm_output_dim = lstm_hidden_size * 2

        self.attention = TemporalAttention(
            hidden_dim=lstm_output_dim,
            attention_dim=attention_dim,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(p=dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.fusion(x)
        x = self.down1(x)
        x = self.down2(x)

        x = x.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]
        lstm_out, _ = self.lstm(x)

        context, _ = self.attention(lstm_out)
        logits = self.classifier(context)

        return logits


if __name__ == "__main__":
    model = MSCNNLSTMAttention(num_classes=10)
    dummy_input = torch.randn(8, 1, 1024)
    dummy_output = model(dummy_input)

    print(model)
    print("Input shape :", dummy_input.shape)
    print("Output shape:", dummy_output.shape)
