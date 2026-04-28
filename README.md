# CWRU Fault Diagnosis

面向变负载工况的轴承故障深度时序建模与多尺度注意力增强方法分析。

当前项目已经完成 Day 1 到 Day 3：

- Day 1：整理 CWRU 40 个 `.mat` 文件，建立 `metadata.csv`，检查 DE time series 是否可读取。
- Day 2：构建防数据泄露的数据处理流程，完成时序切分、滑窗、标准化，并生成训练/测试窗口数据。
- Day 3：实现 1D-CNN baseline，完成 30 epoch 训练闭环，保存日志、曲线图和最佳模型。

## Project Structure

```text
cwru_fault_diagnosis/
├── metadata.csv
├── data/
│   ├── raw_mat/              # 本地原始 CWRU .mat 文件，未上传 GitHub
│   └── processed/
│       └── dataset_info.txt  # 数据集构建信息
├── src/
│   ├── check_mat_files.py    # 检查 .mat 文件和 DE signal key
│   ├── dataset.py            # 读取数据、重采样、时序切分、滑窗
│   ├── make_npz.py           # 生成 train/test npz 数据
│   ├── train_cnn1d.py        # 训练 1D-CNN baseline
│   └── models/
│       └── cnn1d.py          # 1D-CNN 模型定义
└── results/
    ├── logs/                 # 数据检查与训练日志
    ├── figures/              # 训练曲线图
    └── checkpoints/          # 本地模型权重，未上传 GitHub
```

## Dataset

使用 CWRU Bearing Data Center 数据：

- 10 个类别：Normal、IR007、IR014、IR021、B007、B014、B021、OR007@6、OR014@6、OR021@6
- 4 个负载：0 / 1 / 2 / 3 hp
- 每个文件提取 Drive End vibration signal

当前处理参数：

```text
window_size: 1024
stride: 512
train_ratio: 0.7
target_sampling_rate: 12000
X_train shape: (6518, 1, 1024)
X_test shape : (2764, 1, 1024)
```

## Data Leakage Prevention

本项目没有采用“先滑窗再随机划分”的方式，而是对每条原始振动信号先做 chronological split：

```text
前 70% 原始时间序列 -> 训练段 -> 滑窗
后 30% 原始时间序列 -> 测试段 -> 滑窗
```

这样可以避免相邻窗口同时出现在训练集和测试集中造成数据泄露。

## Run

检查原始 `.mat` 文件：

```bash
python src/check_mat_files.py
```

构建窗口数据：

```bash
python src/make_npz.py
```

训练 CNN1D baseline：

```bash
python src/train_cnn1d.py
```

## Current Result

`CNN1D` baseline 已完成 30 epoch 训练。训练日志和曲线保存在：

```text
results/logs/cnn1d_baseline_30epoch_log.csv
results/figures/cnn1d_baseline_30epoch_loss_curve.png
results/figures/cnn1d_baseline_30epoch_metric_curve.png
```

模型权重文件较大，保存在本地 `results/checkpoints/`，未上传到 GitHub。
