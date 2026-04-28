from pathlib import Path
import csv
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw_mat"
METADATA_PATH = PROJECT_ROOT / "metadata.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "logs" / "mat_check_report.csv"


def find_de_key(mat_dict):
    """
    自动寻找 DE time series 对应的 key。
    例如：X097_DE_time、X105_DE_time
    """
    for key in mat_dict.keys():
        if key.startswith("__"):
            continue

        key_upper = key.upper()
        if "DE" in key_upper and "TIME" in key_upper:
            return key

    return None


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    report_rows = []

    print("Project root:", PROJECT_ROOT)
    print("Raw mat dir :", RAW_DIR)
    print("Metadata    :", METADATA_PATH)
    print("Report path :", REPORT_PATH)
    print("-" * 80)
    print(f"Total files in metadata: {len(rows)}")
    print("-" * 80)

    for idx, row in enumerate(rows, start=1):
        filename = row["filename"].strip()
        file_path = RAW_DIR / filename

        print(f"[{idx:02d}] file: {filename}")

        result = {
            "filename": filename,
            "exists": file_path.exists(),
            "label": row["label"],
            "label_name": row["label_name"],
            "fault_type": row["fault_type"],
            "fault_size": row["fault_size"],
            "load_hp": row["load_hp"],
            "motor_speed_rpm": row["motor_speed_rpm"],
            "sampling_rate": row["sampling_rate"],
            "expected_signal_key": row["signal_key"],
            "detected_de_key": "",
            "signal_length": 0,
            "status": "OK",
        }

        if not file_path.exists():
            result["status"] = "FILE_NOT_FOUND"
            report_rows.append(result)
            print("  ERROR: file not found")
            print("-" * 80)
            continue

        mat_data = loadmat(file_path)

        keys = [k for k in mat_data.keys() if not k.startswith("__")]

        expected_key = row["signal_key"].strip()
        if expected_key in mat_data:
            de_key = expected_key
        else:
            de_key = find_de_key(mat_data)

        result["detected_de_key"] = de_key if de_key is not None else ""

        print("  keys               :", keys)
        print("  expected signal key:", expected_key)
        print("  detected DE key    :", de_key)

        if de_key is None:
            result["status"] = "DE_KEY_NOT_FOUND"
            report_rows.append(result)
            print("  ERROR: DE time series not found")
            print("-" * 80)
            continue

        signal = mat_data[de_key].squeeze()
        result["signal_length"] = len(signal)

        print("  signal length      :", len(signal))
        print("  label              :", row["label"])
        print("  label name         :", row["label_name"])
        print("  load hp            :", row["load_hp"])
        print("  status             : OK")
        print("-" * 80)

        report_rows.append(result)

    fieldnames = [
        "filename",
        "exists",
        "label",
        "label_name",
        "fault_type",
        "fault_size",
        "load_hp",
        "motor_speed_rpm",
        "sampling_rate",
        "expected_signal_key",
        "detected_de_key",
        "signal_length",
        "status",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    ok_count = sum(1 for r in report_rows if r["status"] == "OK")
    print(f"Check finished: {ok_count}/{len(report_rows)} files OK")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
