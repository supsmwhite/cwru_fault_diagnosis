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


TRAIN_NPZ = PROJECT_ROOT / "data" / "processed" / "train_windows.npz"
TEST_NPZ = PROJECT_ROOT / "data" / "processed" / "test_windows.npz"

LOG_DIR = PROJECT_ROOT / "results" / "logs"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"

LOG_PATH = LOG_DIR / "resnet1d_baseline_log.csv"
BEST_MODEL_PATH = CHECKPOINT_DIR / "resnet1d_baseline_best.pt"
LOSS_FIG_PATH = FIGURE_DIR / "resnet1d_baseline_loss_curve.png"
METRIC_FIG_PATH = FIGURE_DIR / "resnet1d_baseline_metric_curve.png"

# =========================
# 训练参数：Day 4 ResNet1D baseline
# =========================
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42


def set_seed(seed=42):
    """
    固定随机种子，让每次训练结果尽量可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """
    自动选择 GPU 或 CPU。
    如果 CUDA 可用，就使用 GPU。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_npz_dataset():
    """
    加载 Day 2 保存好的窗口数据。

    X_train: [6518, 1, 1024]
    y_train: [6518]
    X_test : [2764, 1, 1024]
    y_test : [2764]
    """
    train_data = np.load(TRAIN_NPZ)
    test_data = np.load(TEST_NPZ)

    X_train = train_data["X"].astype(np.float32)
    y_train = train_data["y"].astype(np.int64)

    X_test = test_data["X"].astype(np.float32)
    y_test = test_data["y"].astype(np.int64)

    return X_train, y_train, X_test, y_test


def build_dataloaders(X_train, y_train, X_test, y_test):
    """
    把 numpy 数组转成 PyTorch DataLoader。

    DataLoader 的作用：
    - 自动按 batch 取数据；
    - 训练集 shuffle；
    - 测试集不 shuffle。
    """
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


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
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

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
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, macro_f1


def save_log(history):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
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

        for row in history:
            writer.writerow(row)


def save_figures(history):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]

    train_losses = [row["train_loss"] for row in history]
    test_losses = [row["test_loss"] for row in history]

    train_accs = [row["train_acc"] for row in history]
    test_accs = [row["test_acc"] for row in history]
    test_f1s = [row["test_macro_f1"] for row in history]

    plt.figure()
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, test_losses, marker="o", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet1D Baseline 10-Epoch Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_FIG_PATH, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accs, marker="o", label="Train Accuracy")
    plt.plot(epochs, test_accs, marker="o", label="Test Accuracy")
    plt.plot(epochs, test_f1s, marker="o", label="Test Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("ResNet1D Baseline 10-Epoch Metric Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(METRIC_FIG_PATH, dpi=300)
    plt.close()


def main():
    set_seed(SEED)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 80)
    print("ResNet1D baseline 10-epoch training")
    print("=" * 80)
    print("Device:", device)
    print("Train npz:", TRAIN_NPZ)
    print("Test npz :", TEST_NPZ)

    X_train, y_train, X_test, y_test = load_npz_dataset()

    print("-" * 80)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_test shape :", y_test.shape)

    train_loader, test_loader = build_dataloaders(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    model = ResNet1D(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_test_acc = 0.0
    history = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_acc, test_f1 = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_macro_f1": test_f1,
        }

        history.append(row)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": test_acc,
                    "test_macro_f1": test_f1,
                },
                BEST_MODEL_PATH,
            )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"train_f1={train_f1:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"test_f1={test_f1:.4f}"
        )

    elapsed = time.time() - start_time

    save_log(history)
    save_figures(history)

    print("-" * 80)
    print(f"Training finished in {elapsed:.1f} seconds")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print("Log saved to :", LOG_PATH)
    print("Best model saved to:", BEST_MODEL_PATH)
    print("Loss figure saved :", LOSS_FIG_PATH)
    print("Metric figure saved:", METRIC_FIG_PATH)
    print("=" * 80)


if __name__ == "__main__":
    main()