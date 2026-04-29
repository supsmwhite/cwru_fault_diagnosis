from pathlib import Path
import sys
import csv
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
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
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "few_shot_load3_augmented_multiseed"

EPOCH_LOG_PATH = LOG_DIR / "few_shot_load3_augmented_multiseed_epoch_log.csv"
RAW_SUMMARY_PATH = LOG_DIR / "few_shot_load3_augmented_multiseed_raw_summary.csv"
MEAN_STD_PATH = LOG_DIR / "few_shot_load3_augmented_multiseed_mean_std.csv"

ACC_FIG_PATH = FIGURE_DIR / "load3_accuracy_mean_std.png"
F1_FIG_PATH = FIGURE_DIR / "load3_macro_f1_mean_std.png"


# =========================
# 实验设置
# =========================
SOURCE_LOADS = [0, 1, 2]
TARGET_LOAD = 3

SEEDS = [42, 2024, 3407]

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


# =========================
# 数据增强参数
# =========================
USE_AUGMENTATION = True

AMP_SCALE_MIN = 0.8
AMP_SCALE_MAX = 1.2

NOISE_SNR_MIN_DB = 10.0
NOISE_SNR_MAX_DB = 30.0

MAX_TIME_SHIFT = 64


def set_seed(seed):
    """
    固定随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    """
    拼接 Load 0 + Load 1 + Load 2 的训练集。
    """
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
    按类别分层抽样。

    只作用于 source loads 的 train set：
        Load 0 + Load 1 + Load 2

    不作用于：
        source validation
        target Load 3 test
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


class AugmentedWindowDataset(Dataset):
    """
    只用于训练集的数据增强 Dataset。

    输入:
        X: [num_samples, 1, 1024]
        y: [num_samples]

    增强策略:
        1. 幅值缩放
        2. 加性高斯噪声
        3. 随机时间平移

    注意:
        验证集和测试集不能使用该 Dataset。
    """

    def __init__(
        self,
        X,
        y,
        use_augmentation=True,
        amp_scale_min=0.8,
        amp_scale_max=1.2,
        noise_snr_min_db=10.0,
        noise_snr_max_db=30.0,
        max_time_shift=64,
    ):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

        self.use_augmentation = use_augmentation
        self.amp_scale_min = amp_scale_min
        self.amp_scale_max = amp_scale_max
        self.noise_snr_min_db = noise_snr_min_db
        self.noise_snr_max_db = noise_snr_max_db
        self.max_time_shift = max_time_shift

    def __len__(self):
        return len(self.y)

    def random_amplitude_scaling(self, x):
        scale = torch.empty(1).uniform_(
            self.amp_scale_min,
            self.amp_scale_max,
        )
        return x * scale

    def random_gaussian_noise(self, x):
        snr_db = torch.empty(1).uniform_(
            self.noise_snr_min_db,
            self.noise_snr_max_db,
        ).item()

        signal_power = torch.mean(x ** 2)

        if signal_power.item() <= 1e-12:
            return x

        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise_std = torch.sqrt(noise_power)

        noise = torch.randn_like(x) * noise_std

        return x + noise

    def random_time_shift(self, x):
        if self.max_time_shift <= 0:
            return x

        shift = random.randint(-self.max_time_shift, self.max_time_shift)

        if shift == 0:
            return x

        return torch.roll(x, shifts=shift, dims=-1)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.use_augmentation:
            x = self.random_amplitude_scaling(x)
            x = self.random_gaussian_noise(x)
            x = self.random_time_shift(x)

        return x, y


def build_train_loader(X, y, use_augmentation):
    dataset = AugmentedWindowDataset(
        X,
        y,
        use_augmentation=use_augmentation,
        amp_scale_min=AMP_SCALE_MIN,
        amp_scale_max=AMP_SCALE_MAX,
        noise_snr_min_db=NOISE_SNR_MIN_DB,
        noise_snr_max_db=NOISE_SNR_MAX_DB,
        max_time_shift=MAX_TIME_SHIFT,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def build_eval_loader(X, y):
    """
    验证集和测试集不增强。
    """
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def build_source_validation_loaders(source_loads):
    """
    使用 source loads 的 test_windows 作为验证集。

    例如：
        Train Load 0+1+2
        Validation Load 0+1+2 test_windows
        Final Test Load 3 test_windows

    注意：
        Load 3 不参与 validation。
    """
    val_loaders = {}
    val_sample_counts = {}

    for load_hp in source_loads:
        X_val, y_val, _ = load_test_npz(load_hp)
        val_loaders[load_hp] = build_eval_loader(X_val, y_val)
        val_sample_counts[load_hp] = len(y_val)

    return val_loaders, val_sample_counts


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


def evaluate_source_validation(model, val_loaders, criterion, device):
    """
    只用 source loads 的验证集选择 best epoch。
    不使用 target Load 3。
    """
    val_losses = []
    val_accs = []
    val_f1s = []

    for load_hp, loader in val_loaders.items():
        val_loss, val_acc, val_f1 = evaluate(
            model,
            loader,
            criterion,
            device,
        )

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

    mean_val_loss = float(np.mean(val_losses))
    mean_val_acc = float(np.mean(val_accs))
    mean_val_f1 = float(np.mean(val_f1s))

    return mean_val_loss, mean_val_acc, mean_val_f1


def copy_state_dict_to_cpu(model):
    """
    保存当前 best 模型参数到 CPU 内存。
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_one_setting(model_name, train_ratio, seed, device):
    """
    多 seed 严格增强版小样本跨负载实验。

    Train:
        Load 0+1+2 的部分训练样本，训练时增强。

    Validation:
        Load 0+1+2 的 test set，不增强，用于选 best epoch。

    Final Test:
        Load 3 test set，不增强，只在训练完成后测试一次。

    严格性原则:
        Load 3 不参与训练；
        Load 3 不参与 best epoch 选择；
        Load 3 只做最终测试。
    """
    set_seed(seed)

    X_full, y_full, train_paths = build_multi_source_train_data(SOURCE_LOADS)

    X_train, y_train = stratified_subsample(
        X_full,
        y_full,
        ratio=train_ratio,
        seed=seed,
    )

    train_loader = build_train_loader(
        X_train,
        y_train,
        use_augmentation=USE_AUGMENTATION,
    )

    source_val_loaders, source_val_sample_counts = build_source_validation_loaders(
        SOURCE_LOADS
    )

    X_target, y_target, target_test_path = load_test_npz(TARGET_LOAD)
    target_loader = build_eval_loader(X_target, y_target)

    model = MODEL_BUILDERS[model_name]().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("=" * 80)
    print(
        f"MULTI-SEED AUGMENTED few-shot Load3 | "
        f"Model: {model_name} | Ratio: {train_ratio} | Seed: {seed}"
    )
    print("=" * 80)
    print("Device:", device)
    print("Source loads:", SOURCE_LOADS)
    print("Target load:", TARGET_LOAD)
    print("Full train samples:", len(y_full))
    print("Used train samples:", len(y_train))
    print("Source validation samples:", sum(source_val_sample_counts.values()))
    print("Target test samples:", len(y_target))
    print("Use augmentation:", USE_AUGMENTATION)
    print("Amplitude scale:", AMP_SCALE_MIN, "to", AMP_SCALE_MAX)
    print("Noise SNR dB:", NOISE_SNR_MIN_DB, "to", NOISE_SNR_MAX_DB)
    print("Max time shift:", MAX_TIME_SHIFT)
    print("Train paths:")
    for path in train_paths:
        print("  ", path)
    print("Target test path:", target_test_path)
    print("Selection rule: best epoch by source validation mean accuracy, then mean F1")

    best_val_acc = -1.0
    best_val_f1 = -1.0
    best_val_loss = None
    best_epoch = -1
    best_state_dict = None

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

        val_loss, val_acc, val_f1 = evaluate_source_validation(
            model,
            source_val_loaders,
            criterion,
            device,
        )

        row = {
            "seed": seed,
            "model": model_name,
            "train_ratio": train_ratio,
            "epoch": epoch,
            "train_samples": len(y_train),
            "source_loads": "+".join(map(str, SOURCE_LOADS)),
            "target_load": TARGET_LOAD,
            "use_augmentation": USE_AUGMENTATION,
            "amp_scale_min": AMP_SCALE_MIN,
            "amp_scale_max": AMP_SCALE_MAX,
            "noise_snr_min_db": NOISE_SNR_MIN_DB,
            "noise_snr_max_db": NOISE_SNR_MAX_DB,
            "max_time_shift": MAX_TIME_SHIFT,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "source_val_loss": val_loss,
            "source_val_acc": val_acc,
            "source_val_macro_f1": val_f1,
        }

        epoch_rows.append(row)

        is_better = False

        if val_acc > best_val_acc:
            is_better = True
        elif val_acc == best_val_acc and val_f1 > best_val_f1:
            is_better = True

        if is_better:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy_state_dict_to_cpu(model)

        print(
            f"{model_name} | ratio={train_ratio:.1f} | seed={seed} | "
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"train_acc={train_acc:.4f} | "
            f"source_val_acc={val_acc:.4f} | "
            f"source_val_f1={val_f1:.4f}"
        )

    if best_state_dict is None:
        raise RuntimeError("No best model state was saved. Check validation logic.")

    model.load_state_dict(best_state_dict)
    model.to(device)

    target_loss, target_acc, target_f1 = evaluate(
        model,
        target_loader,
        criterion,
        device,
    )

    elapsed = time.time() - start_time

    summary_row = {
        "seed": seed,
        "model": model_name,
        "train_ratio": train_ratio,
        "selected_epoch": best_epoch,
        "train_samples": len(y_train),
        "source_loads": "+".join(map(str, SOURCE_LOADS)),
        "source_val_acc": best_val_acc,
        "source_val_macro_f1": best_val_f1,
        "source_val_loss": best_val_loss,
        "target_load": TARGET_LOAD,
        "target_test_acc": target_acc,
        "target_test_macro_f1": target_f1,
        "target_test_loss": target_loss,
        "elapsed_seconds": elapsed,
        "use_augmentation": USE_AUGMENTATION,
        "amp_scale_min": AMP_SCALE_MIN,
        "amp_scale_max": AMP_SCALE_MAX,
        "noise_snr_min_db": NOISE_SNR_MIN_DB,
        "noise_snr_max_db": NOISE_SNR_MAX_DB,
        "max_time_shift": MAX_TIME_SHIFT,
    }

    print("-" * 80)
    print(f"{model_name} ratio={train_ratio:.1f} seed={seed} finished in {elapsed:.1f} seconds")
    print(f"Selected epoch by source validation: {best_epoch}")
    print(f"Best source-val acc: {best_val_acc:.4f}")
    print(f"Best source-val F1 : {best_val_f1:.4f}")
    print(f"Final Load3 test acc: {target_acc:.4f}")
    print(f"Final Load3 test F1 : {target_f1:.4f}")

    return epoch_rows, summary_row


def save_epoch_log(all_epoch_rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(EPOCH_LOG_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "seed",
            "model",
            "train_ratio",
            "epoch",
            "train_samples",
            "source_loads",
            "target_load",
            "use_augmentation",
            "amp_scale_min",
            "amp_scale_max",
            "noise_snr_min_db",
            "noise_snr_max_db",
            "max_time_shift",
            "train_loss",
            "train_acc",
            "train_macro_f1",
            "source_val_loss",
            "source_val_acc",
            "source_val_macro_f1",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_epoch_rows:
            writer.writerow(row)


def save_raw_summary(raw_summary_rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_SUMMARY_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "seed",
            "model",
            "train_ratio",
            "selected_epoch",
            "train_samples",
            "source_loads",
            "source_val_acc",
            "source_val_macro_f1",
            "source_val_loss",
            "target_load",
            "target_test_acc",
            "target_test_macro_f1",
            "target_test_loss",
            "elapsed_seconds",
            "use_augmentation",
            "amp_scale_min",
            "amp_scale_max",
            "noise_snr_min_db",
            "noise_snr_max_db",
            "max_time_shift",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in raw_summary_rows:
            writer.writerow(row)


def build_mean_std_rows(raw_summary_rows):
    """
    对 3 个 seed 的结果做 mean ± std。
    """
    mean_std_rows = []

    for model_name in MODEL_BUILDERS.keys():
        for ratio in TRAIN_RATIOS:
            rows = [
                row for row in raw_summary_rows
                if row["model"] == model_name and row["train_ratio"] == ratio
            ]

            if len(rows) == 0:
                continue

            acc_values = np.array(
                [row["target_test_acc"] for row in rows],
                dtype=np.float32,
            )

            f1_values = np.array(
                [row["target_test_macro_f1"] for row in rows],
                dtype=np.float32,
            )

            val_acc_values = np.array(
                [row["source_val_acc"] for row in rows],
                dtype=np.float32,
            )

            val_f1_values = np.array(
                [row["source_val_macro_f1"] for row in rows],
                dtype=np.float32,
            )

            selected_epochs = np.array(
                [row["selected_epoch"] for row in rows],
                dtype=np.float32,
            )

            train_samples = rows[0]["train_samples"]

            mean_std_rows.append({
                "model": model_name,
                "train_ratio": ratio,
                "num_seeds": len(rows),
                "train_samples": train_samples,

                "target_acc_mean": float(np.mean(acc_values)),
                "target_acc_std": float(np.std(acc_values, ddof=1)) if len(rows) > 1 else 0.0,

                "target_macro_f1_mean": float(np.mean(f1_values)),
                "target_macro_f1_std": float(np.std(f1_values, ddof=1)) if len(rows) > 1 else 0.0,

                "source_val_acc_mean": float(np.mean(val_acc_values)),
                "source_val_acc_std": float(np.std(val_acc_values, ddof=1)) if len(rows) > 1 else 0.0,

                "source_val_macro_f1_mean": float(np.mean(val_f1_values)),
                "source_val_macro_f1_std": float(np.std(val_f1_values, ddof=1)) if len(rows) > 1 else 0.0,

                "selected_epoch_mean": float(np.mean(selected_epochs)),
                "selected_epoch_std": float(np.std(selected_epochs, ddof=1)) if len(rows) > 1 else 0.0,
            })

    return mean_std_rows


def save_mean_std(mean_std_rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(MEAN_STD_PATH, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "model",
            "train_ratio",
            "num_seeds",
            "train_samples",

            "target_acc_mean",
            "target_acc_std",

            "target_macro_f1_mean",
            "target_macro_f1_std",

            "source_val_acc_mean",
            "source_val_acc_std",

            "source_val_macro_f1_mean",
            "source_val_macro_f1_std",

            "selected_epoch_mean",
            "selected_epoch_std",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in mean_std_rows:
            writer.writerow(row)


def plot_mean_std(mean_std_rows, mean_key, std_key, ylabel, title, save_path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for model_name in MODEL_BUILDERS.keys():
        rows = [
            row for row in mean_std_rows
            if row["model"] == model_name
        ]

        rows = sorted(rows, key=lambda row: row["train_ratio"])

        ratios = [row["train_ratio"] for row in rows]
        means = [row[mean_key] for row in rows]
        stds = [row[std_key] for row in rows]

        plt.errorbar(
            ratios,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=model_name,
        )

        for x, y in zip(ratios, means):
            plt.text(
                x,
                y + 0.01,
                f"{y:.4f}",
                ha="center",
                va="bottom",
            )

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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 80)
    print("MULTI-SEED AUGMENTED strict few-shot cross-load comparison")
    print("Train Load 0+1+2 with augmentation -> Validate Load 0+1+2 -> Final Test Load 3")
    print("=" * 80)
    print("Device:", device)
    print("Seeds:", SEEDS)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Train ratios:", TRAIN_RATIOS)
    print("Models:", list(MODEL_BUILDERS.keys()))
    print("Use augmentation:", USE_AUGMENTATION)

    all_epoch_rows = []
    raw_summary_rows = []

    for seed in SEEDS:
        for model_name in MODEL_BUILDERS.keys():
            for ratio in TRAIN_RATIOS:
                epoch_rows, summary_row = train_one_setting(
                    model_name=model_name,
                    train_ratio=ratio,
                    seed=seed,
                    device=device,
                )

                all_epoch_rows.extend(epoch_rows)
                raw_summary_rows.append(summary_row)

    save_epoch_log(all_epoch_rows)
    save_raw_summary(raw_summary_rows)

    mean_std_rows = build_mean_std_rows(raw_summary_rows)
    save_mean_std(mean_std_rows)

    plot_mean_std(
        mean_std_rows,
        mean_key="target_acc_mean",
        std_key="target_acc_std",
        ylabel="Accuracy",
        title="Multi-seed Augmented STRICT Few-shot Accuracy: Final Test Load 3",
        save_path=ACC_FIG_PATH,
    )

    plot_mean_std(
        mean_std_rows,
        mean_key="target_macro_f1_mean",
        std_key="target_macro_f1_std",
        ylabel="Macro-F1",
        title="Multi-seed Augmented STRICT Few-shot Macro-F1: Final Test Load 3",
        save_path=F1_FIG_PATH,
    )

    print("=" * 80)
    print("MULTI-SEED augmented strict few-shot Load3 comparison finished.")
    print("Epoch log saved to:", EPOCH_LOG_PATH)
    print("Raw summary saved to:", RAW_SUMMARY_PATH)
    print("Mean/std summary saved to:", MEAN_STD_PATH)
    print("Accuracy mean/std figure saved to:", ACC_FIG_PATH)
    print("Macro-F1 mean/std figure saved to:", F1_FIG_PATH)
    print("=" * 80)

    print("Mean ± Std Summary:")
    for row in mean_std_rows:
        print(
            f"{row['model']} | "
            f"ratio={row['train_ratio']:.1f} | "
            f"n={row['num_seeds']} | "
            f"acc={row['target_acc_mean']:.4f} ± {row['target_acc_std']:.4f} | "
            f"f1={row['target_macro_f1_mean']:.4f} ± {row['target_macro_f1_std']:.4f}"
        )


if __name__ == "__main__":
    main()