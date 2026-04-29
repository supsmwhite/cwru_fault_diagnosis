import torch
import torch.nn as nn


def get_group_count(channels, max_groups=8):
    """
    为 GroupNorm 自动选择合适的 group 数。
    作用：
        避免 BatchNorm 在小样本 + 数据增强场景下 running statistics 波动。
    """
    for groups in range(max_groups, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNReLU(nn.Module):
    """
    Conv1d + GroupNorm + ReLU block.

    使用 GroupNorm 而不是 BatchNorm：
        - 对 batch size 不敏感；
        - 在小样本和增强场景下更稳定。
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


class TemporalAttention(nn.Module):
    """
    时间注意力池化。

    Input:
        lstm_out: [batch_size, seq_len, hidden_dim]

    Output:
        context: [batch_size, hidden_dim]
        weights: [batch_size, seq_len]
    """

    def __init__(self, hidden_dim, attention_dim=64):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, lstm_out):
        scores = self.attention(lstm_out).squeeze(-1)      # [B, T]
        weights = torch.softmax(scores, dim=1)             # [B, T]

        context = torch.sum(
            lstm_out * weights.unsqueeze(-1),
            dim=1,
        )                                                  # [B, H]

        return context, weights


class CNNLSTMAttention(nn.Module):
    """
    CNN-BiLSTM-Attention for CWRU bearing fault diagnosis.

    Input:
        x: [batch_size, 1, 1024]

    Output:
        logits: [batch_size, num_classes]

    Model structure:
        1D-CNN feature extractor
        -> downsampled temporal feature sequence
        -> BiLSTM temporal modeling
        -> temporal attention pooling
        -> classifier
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

        self.feature_extractor = nn.Sequential(
            ConvGNReLU(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                stride=2,
                dropout=0.0,
            ),  # [B, 32, 512]

            ConvGNReLU(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                dropout=dropout,
            ),  # [B, 64, 256]

            ConvGNReLU(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dropout=dropout,
            ),  # [B, 128, 128]
        )

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
        x = self.feature_extractor(x)     # [B, 128, 128]

        x = x.permute(0, 2, 1)            # [B, 128, 128] -> [B, T, C]

        lstm_out, _ = self.lstm(x)        # [B, T, 2H]

        context, _ = self.attention(lstm_out)

        logits = self.classifier(context)

        return logits


class CNNBiLSTM(nn.Module):
    """
    Ablation model: CNN-BiLSTM without temporal attention.

    It keeps the same CNN feature extractor and BiLSTM backbone as
    CNNLSTMAttention, but replaces attention pooling with mean pooling.
    This is used to isolate the contribution of temporal attention.
    """

    def __init__(
        self,
        num_classes=10,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        dropout=0.3,
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ConvGNReLU(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                stride=2,
                dropout=0.0,
            ),

            ConvGNReLU(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                dropout=dropout,
            ),

            ConvGNReLU(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dropout=dropout,
            ),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if lstm_num_layers == 1 else dropout,
        )

        lstm_output_dim = lstm_hidden_size * 2

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(p=dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)     # [B, 128, 128]

        x = x.permute(0, 2, 1)            # [B, T, C]

        lstm_out, _ = self.lstm(x)        # [B, T, 2H]

        context = lstm_out.mean(dim=1)    # mean temporal pooling

        logits = self.classifier(context)

        return logits


if __name__ == "__main__":
    model = CNNLSTMAttention(num_classes=10)

    dummy_input = torch.randn(8, 1, 1024)
    dummy_output = model(dummy_input)

    print(model)
    print("Input shape :", dummy_input.shape)
    print("Output shape:", dummy_output.shape)
