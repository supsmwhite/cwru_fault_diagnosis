from pathlib import Path

import numpy as np

import train_few_shot_load3_compare_augmented_multiseed as base

from models.cnn1d import CNN1D
from models.cnn_lstm_attention import CNNLSTMAttention
from models.ms_cnn_lstm_attention import MSCNNLSTMAttention


PROJECT_ROOT = base.PROJECT_ROOT


# =========================
# Output paths
# =========================
base.FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "few_shot_load3_ms_cnn_lstm_attention_multiseed"
)

base.EPOCH_LOG_PATH = (
    base.LOG_DIR
    / "few_shot_load3_ms_cnn_lstm_attention_multiseed_epoch_log.csv"
)

base.RAW_SUMMARY_PATH = (
    base.LOG_DIR
    / "few_shot_load3_ms_cnn_lstm_attention_multiseed_raw_summary.csv"
)

base.MEAN_STD_PATH = (
    base.LOG_DIR
    / "few_shot_load3_ms_cnn_lstm_attention_multiseed_mean_std.csv"
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
# Compare baseline, temporal attention, and multi-scale temporal attention
# =========================
base.MODEL_BUILDERS.clear()

base.MODEL_BUILDERS.update({
    "CNN1D": lambda: CNN1D(num_classes=base.NUM_CLASSES),

    "CNN-BiLSTM-Attention": lambda: CNNLSTMAttention(
        num_classes=base.NUM_CLASSES,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        attention_dim=64,
        dropout=0.3,
    ),

    "MS-CNN-BiLSTM-Attention": lambda: MSCNNLSTMAttention(
        num_classes=base.NUM_CLASSES,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        attention_dim=64,
        dropout=0.3,
    ),
})


# The recurrent models were more stable with a slightly smaller learning rate.
base.LEARNING_RATE = 5e-4


def build_multi_source_train_data_with_load_ids(source_loads):
    """
    Same output contract as base.build_multi_source_train_data, but also stores
    per-window load ids for load-class stratified few-shot sampling.
    """
    X_list = []
    y_list = []
    load_id_list = []
    paths = []

    for load_hp in source_loads:
        X, y, path = base.load_train_npz(load_hp)

        X_list.append(X)
        y_list.append(y)
        load_id_list.append(np.full(len(y), load_hp, dtype=np.int64))
        paths.append(path)

    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)
    load_ids = np.concatenate(load_id_list, axis=0)

    base._LAST_LOAD_IDS_FOR_STRATIFIED_SAMPLING = load_ids

    return X_train, y_train, paths


def load_class_stratified_subsample(X, y, ratio, seed):
    """
    Few-shot sampling stratified by both source load and class label.

    This avoids a hidden issue in very small ratios: class-only sampling can
    accidentally under-represent one source load inside some fault classes.
    """
    if ratio >= 1.0:
        return X, y

    load_ids = getattr(base, "_LAST_LOAD_IDS_FOR_STRATIFIED_SAMPLING", None)

    if load_ids is None or len(load_ids) != len(y):
        raise RuntimeError(
            "Load ids are unavailable for load-class stratified sampling."
        )

    rng = np.random.default_rng(seed)
    selected_indices = []

    for load_hp in base.SOURCE_LOADS:
        for label in range(base.NUM_CLASSES):
            indices = np.where((load_ids == load_hp) & (y == label))[0]

            if len(indices) == 0:
                continue

            rng.shuffle(indices)
            n_select = max(1, int(len(indices) * ratio))
            selected_indices.extend(indices[:n_select].tolist())

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    return X[selected_indices], y[selected_indices]


base.build_multi_source_train_data = build_multi_source_train_data_with_load_ids
base.stratified_subsample = load_class_stratified_subsample


if __name__ == "__main__":
    base.main()
