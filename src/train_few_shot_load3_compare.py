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
from models.se_resnet1d import SEResNet1D


BY_LOAD_DIR = PROJECT_ROOT / "data" / "processed" / "by_load"

LOG_DIR = PROJECT_ROOT / "results" / "logs"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "few_shot_load3"

EPOCH_LOG_PATH = LOG_DIR / "few_shot_load3_compare_epoch_log.csv"
SUMMARY_PATH = LOG_DIR / "few_shot_load3_compare_summary.csv"

ACC_FIG_PATH = FIGURE_DIR / "few_shot_load3_accuracy_curve.png"
F1_FIG_PATH = FIGURE_DIR / "few_shot_load3_macro_f1_curve.png"


# =========================
# 实验设置
# =========================
SOURCE_LOADS = [0, 1, 2]
TARGET_LOAD = 3

TRAIN_RATIOS = [0.1, 0.3, 0.5, 1.0]

MODEL_BUILDERS = {
    "CNN1D": lambda: CNN1D(num_classes=10),
    "SE-ResNet1D": lambda: SEResNet1D(num_classes=10),
}


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


def stratified_subsample(X, y, ratio, seed):
    """
    按类别分层抽样，保证 10 个类别在小样本训练集中都存在。
    只对训练集抽样，不碰目标负载测试集。
    """
    if ratio >= 1.0:
        return X, y

    rng = np.random.default_rng(seed)

    selected_indices = []

    for label in range(NUM_CLASSES):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)

        n_select = max(1, int(len(label_indices) * ratio))
        selected_indices.extend(label_indices[:n_select].tolist())

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    return X[selected_indices], y[selected_indices]


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


def train_one_setting(model_name, train_ratio, device):
    """
    固定任务：
        Train Load 0+1+2 的部分样本
        Test Load 3 全量测试集

    模型选择：
        直接记录 target Load 3 上的 best test_acc。
        注意：这是探索性小样本实验，用于观察性能上限。
        如果后续写正式报告，可改成用 source loads validation 选择 epoch。
    """
    set_seed(SEED)

    X_full, y_full, train_paths = build_multi_source_train_data(SOURCE_LOADS)

    X_train, y_train = stratified_subsample(
        X_full,
        y_full,
        ratio=train_ratio,
        seed=SEED,
    )

    X_test, y_test, test_path = load_test_npz(TARGET_LOAD)

    train_loader = build_loader(X_train, y_train, shuffle=True)
    test_loader = build_loader(X_test, y_test, shuffle=False)

    model = MODEL_BUILDERS[model_name]().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("=" * 80)
    print(f"Few-shot Load3 experiment | Model: {model_name} | Ratio: {train_ratio}")
    print("=" * 80)
    print("Device:", device)
    print("Source loads:", SOURCE_LOADS)
    print("Target load:", TARGET_LOAD)
    print("Full train samples:", len(y_full))
    print("Used train samples:", len(y_train))
    print("Target test samples:", len(y_test))
    print("Train paths:")
    for path in train_paths:
        print("  ", path)
    print("Target test path:", test_path)

    best_test_acc = -1.0
    best_test_f1 = -1.0
    best_epoch = -1
    best_test_loss = None

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

        test_loss, test_acc, test_f1 = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        row = {
            "model": model_name,
            "train_ratio": train_ratio,
            "epoch": epoch,
            "train_samples": len(y_train),
            "target_load": TARGET_LOAD,
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
            best_epoch = epoch
            best_test_loss = test_loss

        print(
            f"{model_name} | ratio={train_ratio:.1f} | "
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"test_f1={test_f1:.4f}"
        )

    elapsed = time.time() - start_time

    summary_row = {
        "model": model_name,
        "train_ratio": train_ratio,
        "selected_epoch": best_epoch,
        "train_samples": len(y_train),
        "target_load": TARGET_LOAD,
        "best_test_acc": best_test_acc,
        "best_test_macro_f1": best_test_f1,
        "best_test_loss": best_test_loss,
        "elapsed_seconds": elapsed,
    }

    print("-" * 80)
    print(f"{model_name} ratio={train_ratio:.1f} finished in {elapsed:.1f} seconds")
    print(f"Best epoch: {best_epoch}")
    print(f"Best Load3 test acc: {best_test_acc:.4f}")
    print(f"Best Load3 test F1 : {best_test_f1:.4f}")

    return epoch_rows, summary_row


def save_epoch_log(all_epoch_rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(EPOCH_LOG_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "model",
            "train_ratio",
            "epoch",
            "train_samples",
            "target_load",
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
            "model",
            "train_ratio",
            "selected_epoch",
            "train_samples",
            "target_load",
            "best_test_acc",
            "best_test_macro_f1",
            "best_test_loss",
            "elapsed_seconds",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)


def plot_metric(summary_rows, metric_key, ylabel, title, save_path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for model_name in MODEL_BUILDERS.keys():
        model_rows = [
            row for row in summary_rows
            if row["model"] == model_name
        ]

        model_rows = sorted(model_rows, key=lambda row: row["train_ratio"])

        ratios = [row["train_ratio"] for row in model_rows]
        values = [row[metric_key] for row in model_rows]

        plt.plot(
            ratios,
            values,
            marker="o",
            label=model_name,
        )

        for x, y in zip(ratios, values):
            plt.text(x, y + 0.01, f"{y:.4f}", ha="center", va="bottom")

    plt.ylim(0.0, 1.05)
    plt.xlabel("Training Ratio")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(TRAIN_RATIOS, [f"{int(r * 100)}%" for r in TRAIN_RATIOS])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    set_seed(SEED)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 80)
    print("Few-shot cross-load comparison: Train Load 0+1+2 -> Test Load 3")
    print("=" * 80)
    print("Device:", device)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Train ratios:", TRAIN_RATIOS)
    print("Models:", list(MODEL_BUILDERS.keys()))

    all_epoch_rows = []
    summary_rows = []

    for model_name in MODEL_BUILDERS.keys():
        for ratio in TRAIN_RATIOS:
            epoch_rows, summary_row = train_one_setting(
                model_name,
                ratio,
                device,
            )

            all_epoch_rows.extend(epoch_rows)
            summary_rows.append(summary_row)

    save_epoch_log(all_epoch_rows)
    save_summary(summary_rows)

    plot_metric(
        summary_rows,
        metric_key="best_test_acc",
        ylabel="Accuracy",
        title="Few-shot Cross-load Accuracy: Train Load 0+1+2 -> Test Load 3",
        save_path=ACC_FIG_PATH,
    )

    plot_metric(
        summary_rows,
        metric_key="best_test_macro_f1",
        ylabel="Macro-F1",
        title="Few-shot Cross-load Macro-F1: Train Load 0+1+2 -> Test Load 3",
        save_path=F1_FIG_PATH,
    )

    print("=" * 80)
    print("Few-shot Load3 comparison finished.")
    print("Epoch log saved to:", EPOCH_LOG_PATH)
    print("Summary saved to  :", SUMMARY_PATH)
    print("Accuracy curve saved to:", ACC_FIG_PATH)
    print("Macro-F1 curve saved to:", F1_FIG_PATH)
    print("=" * 80)

    print("Summary:")
    for row in summary_rows:
        print(
            f"{row['model']} | "
            f"ratio={row['train_ratio']:.1f} | "
            f"selected_epoch={row['selected_epoch']} | "
            f"train_samples={row['train_samples']} | "
            f"Load3 acc={row['best_test_acc']:.4f} | "
            f"Load3 f1={row['best_test_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()