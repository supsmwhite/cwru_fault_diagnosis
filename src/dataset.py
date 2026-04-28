from pathlib import Path
import csv
from collections import Counter

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw_mat"
METADATA_PATH = PROJECT_ROOT / "metadata.csv"

LOG_DIR = PROJECT_ROOT / "results" / "logs"
CLASS_SUMMARY_PATH = LOG_DIR / "window_summary_by_class.csv"
FILE_SUMMARY_PATH = LOG_DIR / "window_summary_by_file.csv"

WINDOW_SIZE = 1024
STRIDE = 512
TRAIN_RATIO = 0.7
TARGET_SAMPLING_RATE = 12000


def read_metadata():
    """
    读取 metadata.csv。
    每一行对应一个 .mat 文件。
    """
    with open(METADATA_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_de_signal(row):
    """
    根据 metadata 中的 signal_key 读取 DE time series。
    """
    filename = row["filename"]
    signal_key = row["signal_key"]
    sampling_rate = int(row["sampling_rate"])

    file_path = RAW_DIR / filename
    mat_data = loadmat(file_path)

    if signal_key not in mat_data:
        raise KeyError(f"{filename}: signal key {signal_key} not found")

    signal = mat_data[signal_key].squeeze().astype(np.float32)

    return signal, sampling_rate


def resample_to_12k(signal, sampling_rate):
    """
    把信号统一到 12kHz。

    原因：
    - Normal 文件是 48kHz；
    - Fault 文件是 12kHz；
    - 如果不统一，模型看到的时间尺度不一致。
    """
    if sampling_rate == TARGET_SAMPLING_RATE:
        return signal

    if sampling_rate == 48000:
        signal = resample_poly(signal, up=1, down=4)
        return signal.astype(np.float32)

    raise ValueError(f"Unsupported sampling rate: {sampling_rate}")


def chronological_split(signal, train_ratio=TRAIN_RATIO):
    """
    按时间顺序切分。

    前 70%：训练段
    后 30%：测试段

    这是为了避免数据泄露。
    """
    split_idx = int(len(signal) * train_ratio)

    train_signal = signal[:split_idx]
    test_signal = signal[split_idx:]

    return train_signal, test_signal


def z_score_normalize(window):
    """
    每个窗口单独做 z-score 标准化。

    作用：
    - 让不同文件的幅值尺度更接近；
    - 让模型更关注波形形状。
    """
    mean = window.mean()
    std = window.std()

    if std < 1e-8:
        return window - mean

    return (window - mean) / std


def sliding_window(signal, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    把一条长信号切成很多长度为 window_size 的小窗口。
    """
    windows = []

    max_start = len(signal) - window_size

    if max_start < 0:
        return np.empty((0, window_size), dtype=np.float32)

    for start in range(0, max_start + 1, stride):
        window = signal[start:start + window_size]
        window = z_score_normalize(window)
        windows.append(window)

    return np.array(windows, dtype=np.float32)


def build_windows():
    """
    构建训练窗口和测试窗口。

    返回：
    X_train: [num_train_windows, 1, window_size]
    y_train: [num_train_windows]
    X_test : [num_test_windows, 1, window_size]
    y_test : [num_test_windows]
    """
    rows = read_metadata()

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    detail_rows = []

    for row in rows:
        filename = row["filename"]
        label = int(row["label"])
        label_name = row["label_name"]
        load_hp = int(row["load_hp"])

        signal, sampling_rate = load_de_signal(row)
        signal = resample_to_12k(signal, sampling_rate)

        train_signal, test_signal = chronological_split(signal)

        train_windows = sliding_window(train_signal)
        test_windows = sliding_window(test_signal)

        X_train_list.append(train_windows)
        y_train_list.append(np.full(len(train_windows), label, dtype=np.int64))

        X_test_list.append(test_windows)
        y_test_list.append(np.full(len(test_windows), label, dtype=np.int64))

        detail_rows.append({
            "filename": filename,
            "label": label,
            "label_name": label_name,
            "load_hp": load_hp,
            "original_sampling_rate": sampling_rate,
            "target_sampling_rate": TARGET_SAMPLING_RATE,
            "resampled_length": len(signal),
            "train_windows": len(train_windows),
            "test_windows": len(test_windows),
        })

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    X_train = X_train[:, None, :]
    X_test = X_test[:, None, :]

    return X_train, y_train, X_test, y_test, detail_rows


def save_summaries(y_train, y_test, detail_rows):
    """
    保存两个 CSV 检查报告：
    1. 按类别统计窗口数量；
    2. 按文件统计窗口数量。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_counter = Counter(y_train.tolist())
    test_counter = Counter(y_test.tolist())

    label_to_name = {}
    for row in detail_rows:
        label_to_name[row["label"]] = row["label_name"]

    with open(CLASS_SUMMARY_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "label",
            "label_name",
            "train_windows",
            "test_windows",
            "total_windows",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for label in sorted(label_to_name.keys()):
            train_count = train_counter[label]
            test_count = test_counter[label]

            writer.writerow({
                "label": label,
                "label_name": label_to_name[label],
                "train_windows": train_count,
                "test_windows": test_count,
                "total_windows": train_count + test_count,
            })

    with open(FILE_SUMMARY_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "filename",
            "label",
            "label_name",
            "load_hp",
            "original_sampling_rate",
            "target_sampling_rate",
            "resampled_length",
            "train_windows",
            "test_windows",
            "total_windows",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in detail_rows:
            writer.writerow({
                "filename": row["filename"],
                "label": row["label"],
                "label_name": row["label_name"],
                "load_hp": row["load_hp"],
                "original_sampling_rate": row["original_sampling_rate"],
                "target_sampling_rate": row["target_sampling_rate"],
                "resampled_length": row["resampled_length"],
                "train_windows": row["train_windows"],
                "test_windows": row["test_windows"],
                "total_windows": row["train_windows"] + row["test_windows"],
            })


def print_summary(X_train, y_train, X_test, y_test, detail_rows):
    """
    打印数据集检查信息。
    """
    print("=" * 80)
    print("Dataset summary")
    print("=" * 80)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_test shape :", y_test.shape)

    print("-" * 80)
    print("Train windows per class:")
    train_counter = Counter(y_train.tolist())
    for label in sorted(train_counter):
        print(f"  label {label}: {train_counter[label]}")

    print("-" * 80)
    print("Test windows per class:")
    test_counter = Counter(y_test.tolist())
    for label in sorted(test_counter):
        print(f"  label {label}: {test_counter[label]}")

    print("-" * 80)
    print("Windows per file:")
    for row in detail_rows:
        print(
            f"{row['filename']:>7} | "
            f"label={row['label']} {row['label_name']:<8} | "
            f"load={row['load_hp']} | "
            f"fs={row['original_sampling_rate']} -> {row['target_sampling_rate']} | "
            f"length={row['resampled_length']} | "
            f"train={row['train_windows']} | "
            f"test={row['test_windows']}"
        )

    print("-" * 80)
    print(f"Class summary saved to: {CLASS_SUMMARY_PATH}")
    print(f"File summary saved to : {FILE_SUMMARY_PATH}")
    print("=" * 80)


def main():
    X_train, y_train, X_test, y_test, detail_rows = build_windows()
    save_summaries(y_train, y_test, detail_rows)
    print_summary(X_train, y_train, X_test, y_test, detail_rows)


if __name__ == "__main__":
    main()
