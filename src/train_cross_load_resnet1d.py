from pathlib import Path
import sys
import csv
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =========================
# 路径设置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from models.resnet1d import ResNet1D


BY_LOAD_DIR = PROJECT_ROOT / "data" / "processed" / "by_load"

LOG_DIR = PROJECT_ROOT / "results" / "logs"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "cross_load"

EPOCH_LOG_PATH = LOG_DIR / "cross_load_resnet1d_epoch_log.csv"
SUMMARY_PATH = LOG_DIR / "cross_load_resnet1d_summary.csv"

ACC_MATRIX_FIG_PATH = FIGURE_DIR / "resnet1d_cross_load_accuracy_matrix.png"
F1_MATRIX_FIG_PATH = FIGURE_DIR / "resnet1d_cross_load_macro_f1_matrix.png"

LOADS = [0, 1, 2, 3]


# =========================
# 训练参数
# =========================
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_train_npz(load_hp):
    train_npz = BY_LOAD_DIR / f"load_{load_hp}_train_windows.npz"
    data = np.load(train_npz)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    return X, y, train_npz


def load_test_npz(load_hp):
    test_npz = BY_LOAD_DIR / f"load_{load_hp}_test_windows.npz"
    data = np.load(test_npz)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    return X, y, test_npz


def build_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    iterator = train_loader
    if tqdm is not None:
        iterator = tqdm(train_loader, desc="Training", leave=False)

    for X_batch, y_batch in iterator:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(y_batch.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(train_loader.dataset)

    acc = accuracy_score(all_labels, all_preds)

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    return avg_loss, acc, macro_f1


def evaluate(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(y_batch.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(test_loader.dataset)

    acc = accuracy_score(all_labels, all_preds)

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    return avg_loss, acc, macro_f1


def plot_matrix(matrix, title, save_path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    labels = [f"Load {load}" for load in LOADS]

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, vmin=0.0, vmax=1.0)
    plt.colorbar()

    plt.xticks(range(len(LOADS)), labels)
    plt.yticks(range(len(LOADS)), labels)

    plt.xlabel("Test Load")
    plt.ylabel("Train Load")
    plt.title(title)

    for i in range(len(LOADS)):
        for j in range(len(LOADS)):
            plt.text(
                j,
                i,
                f"{matrix[i, j]:.4f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def train_source_load(source_load, device):
    """
    在 source_load 上训练一个 ResNet1D。
    每个 epoch 结束后，在 Load 0/1/2/3 的 test set 上全部评估。

    模型选择策略：
    - 用 source_load 自身 test_acc 选择 best epoch；
    - 再记录该 epoch 下对所有 target load 的表现；
    - 不使用目标负载结果反向选择模型，避免跨负载评估不公平。
    """
    set_seed(SEED)

    X_train, y_train, train_npz = load_train_npz(source_load)
    train_loader = build_loader(X_train, y_train, shuffle=True)

    test_loaders = {}
    test_sample_counts = {}

    for target_load in LOADS:
        X_test, y_test, test_npz = load_test_npz(target_load)
        test_loaders[target_load] = build_loader(X_test, y_test, shuffle=False)
        test_sample_counts[target_load] = len(y_test)

    model = ResNet1D(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("=" * 80)
    print(f"Cross-load ResNet1D | Train Load {source_load} hp")
    print("=" * 80)
    print("Device:", device)
    print("Train npz:", train_npz)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    best_source_acc = -1.0
    best_epoch = -1
    best_target_results = None

    epoch_rows = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        current_target_results = {}

        for target_load in LOADS:
            test_loss, test_acc, test_f1 = evaluate(
                model,
                test_loaders[target_load],
                criterion,
                device,
            )

            current_target_results[target_load] = {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_macro_f1": test_f1,
            }

            epoch_rows.append({
                "source_load": source_load,
                "target_load": target_load,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_macro_f1": train_f1,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_macro_f1": test_f1,
            })

        source_acc = current_target_results[source_load]["test_acc"]

        if source_acc > best_source_acc:
            best_source_acc = source_acc
            best_epoch = epoch
            best_target_results = current_target_results

        result_text = " | ".join(
            [
                f"T{target_load}: acc={current_target_results[target_load]['test_acc']:.4f}"
                for target_load in LOADS
            ]
        )

        print(
            f"Train Load {source_load} | "
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"train_acc={train_acc:.4f} | "
            f"source_test_acc={source_acc:.4f} | "
            f"{result_text}"
        )

    elapsed = time.time() - start_time

    summary_rows = []

    for target_load in LOADS:
        result = best_target_results[target_load]

        summary_rows.append({
            "source_load": source_load,
            "target_load": target_load,
            "selected_epoch": best_epoch,
            "test_acc": result["test_acc"],
            "test_macro_f1": result["test_macro_f1"],
            "test_loss": result["test_loss"],
            "train_samples": len(y_train),
            "test_samples": test_sample_counts[target_load],
            "elapsed_seconds_for_source_training": elapsed,
        })

    print("-" * 80)
    print(f"Train Load {source_load} finished in {elapsed:.1f} seconds")
    print(f"Selected epoch by source-load test accuracy: {best_epoch}")
    print(f"Best source-load test accuracy: {best_source_acc:.4f}")

    return epoch_rows, summary_rows


def save_epoch_log(all_epoch_rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(EPOCH_LOG_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "source_load",
            "target_load",
            "epoch",
            "train_loss",
            "train_acc",
            "train_macro_f1",
            "test_loss",
            "test_acc",
            "test_macro_f1",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_epoch_rows:
            writer.writerow(row)


def save_summary(summary_rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "source_load",
            "target_load",
            "selected_epoch",
            "test_acc",
            "test_macro_f1",
            "test_loss",
            "train_samples",
            "test_samples",
            "elapsed_seconds_for_source_training",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)


def build_metric_matrices(summary_rows):
    acc_matrix = np.zeros((len(LOADS), len(LOADS)), dtype=np.float32)
    f1_matrix = np.zeros((len(LOADS), len(LOADS)), dtype=np.float32)

    load_to_idx = {load: idx for idx, load in enumerate(LOADS)}

    for row in summary_rows:
        i = load_to_idx[row["source_load"]]
        j = load_to_idx[row["target_load"]]

        acc_matrix[i, j] = row["test_acc"]
        f1_matrix[i, j] = row["test_macro_f1"]

    return acc_matrix, f1_matrix


def main():
    set_seed(SEED)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 80)
    print("Day 7: ResNet1D cross-load generalization experiments")
    print("=" * 80)
    print("Device:", device)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)

    all_epoch_rows = []
    all_summary_rows = []

    for source_load in LOADS:
        epoch_rows, summary_rows = train_source_load(source_load, device)
        all_epoch_rows.extend(epoch_rows)
        all_summary_rows.extend(summary_rows)

    save_epoch_log(all_epoch_rows)
    save_summary(all_summary_rows)

    acc_matrix, f1_matrix = build_metric_matrices(all_summary_rows)

    plot_matrix(
        acc_matrix,
        "ResNet1D Cross-Load Accuracy Matrix",
        ACC_MATRIX_FIG_PATH,
    )

    plot_matrix(
        f1_matrix,
        "ResNet1D Cross-Load Macro-F1 Matrix",
        F1_MATRIX_FIG_PATH,
    )

    print("=" * 80)
    print("Cross-load ResNet1D experiments finished.")
    print("Epoch log saved to:", EPOCH_LOG_PATH)
    print("Summary saved to  :", SUMMARY_PATH)
    print("Accuracy matrix saved to:", ACC_MATRIX_FIG_PATH)
    print("Macro-F1 matrix saved to:", F1_MATRIX_FIG_PATH)
    print("=" * 80)

    print("Accuracy matrix:")
    print(acc_matrix)

    print("Macro-F1 matrix:")
    print(f1_matrix)


if __name__ == "__main__":
    main()