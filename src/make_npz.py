from pathlib import Path
import numpy as np

from dataset import (
    build_windows,
    WINDOW_SIZE,
    STRIDE,
    TRAIN_RATIO,
    TARGET_SAMPLING_RATE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train_windows.npz"
TEST_PATH = PROCESSED_DIR / "test_windows.npz"
INFO_PATH = PROCESSED_DIR / "dataset_info.txt"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Building windows from raw .mat files...")
    X_train, y_train, X_test, y_test, detail_rows = build_windows()

    print("Saving train dataset...")
    np.savez_compressed(
        TRAIN_PATH,
        X=X_train,
        y=y_train,
    )

    print("Saving test dataset...")
    np.savez_compressed(
        TEST_PATH,
        X=X_test,
        y=y_test,
    )

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        f.write("CWRU processed window dataset\n")
        f.write("=" * 40 + "\n")
        f.write(f"window_size: {WINDOW_SIZE}\n")
        f.write(f"stride: {STRIDE}\n")
        f.write(f"train_ratio: {TRAIN_RATIO}\n")
        f.write(f"target_sampling_rate: {TARGET_SAMPLING_RATE}\n")
        f.write(f"X_train shape: {X_train.shape}\n")
        f.write(f"y_train shape: {y_train.shape}\n")
        f.write(f"X_test shape : {X_test.shape}\n")
        f.write(f"y_test shape : {y_test.shape}\n")
        f.write(f"num_files: {len(detail_rows)}\n")

    print("-" * 80)
    print("Processed datasets saved:")
    print("Train:", TRAIN_PATH)
    print("Test :", TEST_PATH)
    print("Info :", INFO_PATH)
    print("-" * 80)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_test shape :", y_test.shape)


if __name__ == "__main__":
    main()
