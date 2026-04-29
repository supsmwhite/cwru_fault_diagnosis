from pathlib import Path

import train_few_shot_load3_compare_augmented_multiseed as base

from models.cnn1d import CNN1D
from models.cnn_lstm_attention import CNNBiLSTM, CNNLSTMAttention


PROJECT_ROOT = base.PROJECT_ROOT


# =========================
# Output paths
# =========================
base.FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "few_shot_load3_cnn_lstm_ablation_multiseed"
)

base.EPOCH_LOG_PATH = (
    base.LOG_DIR
    / "few_shot_load3_cnn_lstm_ablation_multiseed_epoch_log.csv"
)

base.RAW_SUMMARY_PATH = (
    base.LOG_DIR
    / "few_shot_load3_cnn_lstm_ablation_multiseed_raw_summary.csv"
)

base.MEAN_STD_PATH = (
    base.LOG_DIR
    / "few_shot_load3_cnn_lstm_ablation_multiseed_mean_std.csv"
)

base.ACC_FIG_PATH = (
    base.FIGURE_DIR
    / "load3_accuracy_mean_std.png"
)

base.F1_FIG_PATH = (
    base.FIGURE_DIR
    / "load3_macro_f1_mean_std.png"
)


# =========================
# Ablation: CNN only vs CNN-BiLSTM vs CNN-BiLSTM-Attention
# =========================
base.MODEL_BUILDERS.clear()

base.MODEL_BUILDERS.update({
    "CNN1D": lambda: CNN1D(num_classes=base.NUM_CLASSES),

    "CNN-BiLSTM": lambda: CNNBiLSTM(
        num_classes=base.NUM_CLASSES,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        dropout=0.3,
    ),

    "CNN-BiLSTM-Attention": lambda: CNNLSTMAttention(
        num_classes=base.NUM_CLASSES,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        attention_dim=64,
        dropout=0.3,
    ),
})


# Keep the same learning rate used by the previous LSTM-attention comparison.
base.LEARNING_RATE = 5e-4


if __name__ == "__main__":
    base.main()
