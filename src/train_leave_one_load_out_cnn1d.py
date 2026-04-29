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

from models.cnn1d import CNN1D


BY_LOAD_DIR = PROJECT_ROOT / "data" / "processed" / "by_load"

LOG_DIR = PROJECT_ROOT / "results" / "logs"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "leave_one_load_out"

EPOCH_LOG_PATH = LOG_DIR / "leave_one_load_out_cnn1d_epoch_log.csv"
SUMMARY_PATH = LOG_DIR / "leave_one_load_out_cnn1d_summary.csv"

ACC_BAR_FIG_PATH = FIGURE_DIR / "cnn1d_leave_one_load_out_accuracy_bar.png"
F1_BAR_FIG_PATH = FIGURE_DIR / "cnn1d_leave_one_load_out_macro_f1_bar.png"

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


def build_multi_source_train_data(source_loads):
    X_list = []
    y_list = []
    paths = []

    for load_hp in source_loads:
        X, y, path = load_train_npz(load_hp)
        X_list.append(X)
        y_list.append(y)
        paths.append(path)

    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)

    return X_train, y_train, paths


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


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(y_batch.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)

    acc = accuracy_score(all_labels, all_preds)

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    return avg_loss, acc, macro_f1


def train_leave_one_load_out(target_load, device):
    """
    Leave-one-load-out 实验：

    训练：
        使用除 target_load 外的 3 个负载 train set

    模型选择：
        使用源负载自身 test set 的平均 accuracy 选择 best epoch

    最终评估：
        在 target_load 的 test set 上评估

    注意：
        target_load 不参与训练，也不参与 best epoch 选择。
    """
    set_seed(SEED)

    source_loads = [load for load in LOADS if load != target_load]

    X_train, y_train, train_paths = build_multi_source_train_data(source_loads)
    train_loader = build_loader(X_train, y_train, shuffle=True)

    source_val_loaders = {}
    source_val_sample_counts = {}

    for source_load in source_loads:
        X_val, y_val, _ = load_test_npz(source_load)
        source_val_loaders[source_load] = build_loader(X_val, y_val, shuffle=False)
        source_val_sample_counts[source_load] = len(y_val)

    X_target, y_target, target_test_path = load_test_npz(target_load)
    target_loader = build_loader(X_target, y_target, shuffle=False)

    model = CNN1D(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("=" * 80)
    print(f"CNN1D leave-one-load-out | Test Load {target_load} hp")
    print("=" * 80)
    print("Device:", device)
    print("Source loads:", source_loads)
    print("Target load:", target_load)
    print("Train samples:", len(y_train))
    print("Target test samples:", len(y_target))
    print("Train npz paths:")
    for path in train_paths:
        print("  ", path)
    print("Target test npz:", target_test_path)

    best_source_val_acc = -1.0
    best_epoch = -1
    best_target_result = None
    best_source_val_f1 = None

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

        source_val_accs = []
        source_val_f1s = []

        for source_load, source_val_loader in source_val_loaders.items():
            val_loss, val_acc, val_f1 = evaluate(
                model,
                source_val_loader,
                criterion,
                device,
            )

            source_val_accs.append(val_acc)
            source_val_f1s.append(val_f1)

        mean_source_val_acc = float(np.mean(source_val_accs))
        mean_source_val_f1 = float(np.mean(source_val_f1s))

        target_loss, target_acc, target_f1 = evaluate(
            model,
            target_loader,
            criterion,
            device,
        )

        row = {
            "target_load": target_load,
            "source_loads": "+".join(map(str, source_loads)),
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "mean_source_val_acc": mean_source_val_acc,
            "mean_source_val_macro_f1": mean_source_val_f1,
            "target_test_loss": target_loss,
            "target_test_acc": target_acc,
            "target_test_macro_f1": target_f1,
        }

        epoch_rows.append(row)

        if mean_source_val_acc > best_source_val_acc:
            best_source_val_acc = mean_source_val_acc
            best_source_val_f1 = mean_source_val_f1
            best_epoch = epoch

            best_target_result = {
                "target_test_loss": target_loss,
                "target_test_acc": target_acc,
                "target_test_macro_f1": target_f1,
            }

        print(
            f"Target Load {target_load} | "
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"train_acc={train_acc:.4f} | "
            f"source_val_acc={mean_source_val_acc:.4f} | "
            f"target_acc={target_acc:.4f} | "
            f"target_f1={target_f1:.4f}"
        )

    elapsed = time.time() - start_time

    summary_row = {
        "target_load": target_load,
        "source_loads": "+".join(map(str, source_loads)),
        "selected_epoch": best_epoch,
        "mean_source_val_acc": best_source_val_acc,
        "mean_source_val_macro_f1": best_source_val_f1,
        "target_test_acc": best_target_result["target_test_acc"],
        "target_test_macro_f1": best_target_result["target_test_macro_f1"],
        "target_test_loss": best_target_result["target_test_loss"],
        "train_samples": len(y_train),
        "target_test_samples": len(y_target),
        "elapsed_seconds": elapsed,
    }

    print("-" * 80)
    print(f"Target Load {target_load} finished in {elapsed:.1f} seconds")
    print(f"Selected epoch by mean source-load validation accuracy: {best_epoch}")
    print(f"Best mean source-val accuracy: {best_source_val_acc:.4f}")
    print(f"Target test accuracy at selected epoch: {best_target_result['target_test_acc']:.4f}")
    print(f"Target test macro-F1  at selected epoch: {best_target_result['target_test_macro_f1']:.4f}")

    return epoch_rows, summary_row


def save_epoch_log(all_epoch_rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(EPOCH_LOG_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "target_load",
            "source_loads",
            "epoch",
            "train_loss",
            "train_acc",
            "train_macro_f1",
            "mean_source_val_acc",
            "mean_source_val_macro_f1",
            "target_test_loss",
            "target_test_acc",
            "target_test_macro_f1",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_epoch_rows:
            writer.writerow(row)


def save_summary(summary_rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "target_load",
            "source_loads",
            "selected_epoch",
            "mean_source_val_acc",
            "mean_source_val_macro_f1",
            "target_test_acc",
            "target_test_macro_f1",
            "target_test_loss",
            "train_samples",
            "target_test_samples",
            "elapsed_seconds",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)


def plot_bar(summary_rows, metric_key, title, ylabel, save_path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    target_loads = [row["target_load"] for row in summary_rows]
    values = [row[metric_key] for row in summary_rows]

    x_labels = [f"Test Load {load}" for load in target_loads]

    plt.figure(figsize=(8, 5))
    plt.bar(x_labels, values)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Held-out Target Load")
    plt.ylabel(ylabel)
    plt.title(title)

    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    set_seed(SEED)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 80)
    print("CNN1D leave-one-load-out experiments")
    print("=" * 80)
    print("Device:", device)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)

    all_epoch_rows = []
    summary_rows = []

    for target_load in LOADS:
        epoch_rows, summary_row = train_leave_one_load_out(
            target_load,
            device,
        )

        all_epoch_rows.extend(epoch_rows)
        summary_rows.append(summary_row)

    save_epoch_log(all_epoch_rows)
    save_summary(summary_rows)

    plot_bar(
        summary_rows,
        metric_key="target_test_acc",
        title="CNN1D Leave-One-Load-Out Accuracy",
        ylabel="Accuracy",
        save_path=ACC_BAR_FIG_PATH,
    )

    plot_bar(
        summary_rows,
        metric_key="target_test_macro_f1",
        title="CNN1D Leave-One-Load-Out Macro-F1",
        ylabel="Macro-F1",
        save_path=F1_BAR_FIG_PATH,
    )

    print("=" * 80)
    print("CNN1D leave-one-load-out experiments finished.")
    print("Epoch log saved to:", EPOCH_LOG_PATH)
    print("Summary saved to  :", SUMMARY_PATH)
    print("Accuracy bar saved to:", ACC_BAR_FIG_PATH)
    print("Macro-F1 bar saved to:", F1_BAR_FIG_PATH)
    print("=" * 80)

    print("Summary:")
    for row in summary_rows:
        print(
            f"Test Load {row['target_load']} | "
            f"Train Loads {row['source_loads']} | "
            f"selected_epoch={row['selected_epoch']} | "
            f"acc={row['target_test_acc']:.4f} | "
            f"f1={row['target_test_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()