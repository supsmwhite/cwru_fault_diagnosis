from pathlib import Path
import csv
from collections import Counter

import numpy as np

from dataset import (
    read_metadata,
    load_de_signal,
    resample_to_12k,
    chronological_split,
    sliding_window,
    WINDOW_SIZE,
    STRIDE,
    TRAIN_RATIO,
    TARGET_SAMPLING_RATE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "by_load"
LOG_DIR = PROJECT_ROOT / "results" / "logs"

SUMMARY_PATH = LOG_DIR / "window_summary_by_load.csv"

LOADS = [0, 1, 2, 3]


def build_windows_for_load(target_load):
    """
    只构建某一个负载下的数据集。

    仍然坚持 Day 2 的正确流程：
    原始长信号 -> 统一采样率 -> chronological split -> 分别滑窗 -> 保存 npz
    """
    rows = read_metadata()
    selected_rows = [
        row for row in rows
        if int(row["load_hp"]) == target_load
    ]

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    detail_rows = []

    for row in selected_rows:
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


def save_npz_for_load(load_hp, X_train, y_train, X_test, y_test):
    """
    保存某个负载下的 train/test npz。
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / f"load_{load_hp}_train_windows.npz"
    test_path = PROCESSED_DIR / f"load_{load_hp}_test_windows.npz"

    np.savez_compressed(
        train_path,
        X=X_train,
        y=y_train,
    )

    np.savez_compressed(
        test_path,
        X=X_test,
        y=y_test,
    )

    return train_path, test_path


def save_summary(all_detail_rows):
    """
    保存按负载、按文件统计的窗口数量。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "load_hp",
            "filename",
            "label",
            "label_name",
            "original_sampling_rate",
            "target_sampling_rate",
            "resampled_length",
            "train_windows",
            "test_windows",
            "total_windows",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_detail_rows:
            writer.writerow({
                "load_hp": row["load_hp"],
                "filename": row["filename"],
                "label": row["label"],
                "label_name": row["label_name"],
                "original_sampling_rate": row["original_sampling_rate"],
                "target_sampling_rate": row["target_sampling_rate"],
                "resampled_length": row["resampled_length"],
                "train_windows": row["train_windows"],
                "test_windows": row["test_windows"],
                "total_windows": row["train_windows"] + row["test_windows"],
            })


def print_load_summary(load_hp, X_train, y_train, X_test, y_test, train_path, test_path):
    """
    打印某个负载下的数据集检查信息。
    """
    train_counter = Counter(y_train.tolist())
    test_counter = Counter(y_test.tolist())

    print("-" * 80)
    print(f"Load {load_hp} hp dataset")
    print("-" * 80)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_test shape :", y_test.shape)

    print("Train windows per class:")
    for label in sorted(train_counter.keys()):
        print(f"  label {label}: {train_counter[label]}")

    print("Test windows per class:")
    for label in sorted(test_counter.keys()):
        print(f"  label {label}: {test_counter[label]}")

    print("Saved train:", train_path)
    print("Saved test :", test_path)


def main():
    print("=" * 80)
    print("Building same-load datasets")
    print("=" * 80)
    print(f"window_size: {WINDOW_SIZE}")
    print(f"stride: {STRIDE}")
    print(f"train_ratio: {TRAIN_RATIO}")
    print(f"target_sampling_rate: {TARGET_SAMPLING_RATE}")

    all_detail_rows = []

    for load_hp in LOADS:
        X_train, y_train, X_test, y_test, detail_rows = build_windows_for_load(load_hp)

        train_path, test_path = save_npz_for_load(
            load_hp,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        print_load_summary(
            load_hp,
            X_train,
            y_train,
            X_test,
            y_test,
            train_path,
            test_path,
        )

        all_detail_rows.extend(detail_rows)

    save_summary(all_detail_rows)

    print("-" * 80)
    print("Same-load datasets finished.")
    print("Summary saved to:", SUMMARY_PATH)
    print("=" * 80)


if __name__ == "__main__":
    main()