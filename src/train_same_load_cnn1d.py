from pathlib import Path
import sys
import csv
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "same_load_confusion_matrices"

EPOCH_LOG_PATH = LOG_DIR / "same_load_cnn1d_epoch_log.csv"
SUMMARY_PATH = LOG_DIR / "same_load_cnn1d_summary.csv"

LOADS = [0, 1, 2, 3]

LABEL_NAMES = [
    "Normal",
    "IR007",
    "IR014",
    "IR021",
    "B007",
    "B014",
    "B021",
    "OR007@6",
    "OR014@6",
    "OR021@6",
]


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


def load_npz_dataset(load_hp):
    train_npz = BY_LOAD_DIR / f"load_{load_hp}_train_windows.npz"
    test_npz = BY_LOAD_DIR / f"load_{load_hp}_test_windows.npz"

    train_data = np.load(train_npz)
    test_data = np.load(test_npz)

    X_train = train_data["X"].astype(np.float32)
    y_train = train_data["y"].astype(np.int64)

    X_test = test_data["X"].astype(np.float32)
    y_test = test_data["y"].astype(np.int64)

    return X_train, y_train, X_test, y_test, train_npz, test_npz


def build_dataloaders(X_train, y_train, X_test, y_test):
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

    return avg_loss, acc, macro_f1, np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(load_hp, y_true, y_pred):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    fig_path = FIGURE_DIR / f"cnn1d_same_load_load_{load_hp}_confusion_matrix.png"

    plt.figure(figsize=(9, 7))
    plt.imshow(cm)
    plt.title(f"CNN1D Same-Load Confusion Matrix - Load {load_hp} hp")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(NUM_CLASSES), LABEL_NAMES, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), LABEL_NAMES)
    plt.colorbar()

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return fig_path


def train_one_load(load_hp, device):
    set_seed(SEED)

    X_train, y_train, X_test, y_test, train_npz, test_npz = load_npz_dataset(load_hp)

    print("=" * 80)
    print(f"CNN1D same-load training | Load {load_hp} hp")
    print("=" * 80)
    print("Device:", device)
    print("Train npz:", train_npz)
    print("Test npz :", test_npz)
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

    model = CNN1D(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_y_true = None
    best_y_pred = None

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

        test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        row = {
            "load_hp": load_hp,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_macro_f1": test_f1,
        }
        epoch_rows.append(row)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_f1 = test_f1
            best_y_true = y_true
            best_y_pred = y_pred

        print(
            f"Load {load_hp} | "
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"train_f1={train_f1:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"test_f1={test_f1:.4f}"
        )

    elapsed = time.time() - start_time

    cm_path = save_confusion_matrix(
        load_hp,
        best_y_true,
        best_y_pred,
    )

    summary_row = {
        "load_hp": load_hp,
        "best_test_acc": best_test_acc,
        "best_test_macro_f1": best_test_f1,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "elapsed_seconds": elapsed,
        "confusion_matrix_path": str(cm_path),
    }

    print("-" * 80)
    print(f"Load {load_hp} finished in {elapsed:.1f} seconds")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Best test macro-F1 : {best_test_f1:.4f}")
    print("Confusion matrix saved to:", cm_path)

    return epoch_rows, summary_row


def save_epoch_log(all_epoch_rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(EPOCH_LOG_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "load_hp",
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
            "load_hp",
            "best_test_acc",
            "best_test_macro_f1",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "train_samples",
            "test_samples",
            "elapsed_seconds",
            "confusion_matrix_path",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)


def main():
    set_seed(SEED)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 80)
    print("Day 5: CNN1D same-load experiments")
    print("=" * 80)
    print("Device:", device)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)

    all_epoch_rows = []
    summary_rows = []

    for load_hp in LOADS:
        epoch_rows, summary_row = train_one_load(load_hp, device)
        all_epoch_rows.extend(epoch_rows)
        summary_rows.append(summary_row)

    save_epoch_log(all_epoch_rows)
    save_summary(summary_rows)

    print("=" * 80)
    print("Same-load CNN1D experiments finished.")
    print("Epoch log saved to:", EPOCH_LOG_PATH)
    print("Summary saved to  :", SUMMARY_PATH)
    print("Confusion matrices saved to:", FIGURE_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()