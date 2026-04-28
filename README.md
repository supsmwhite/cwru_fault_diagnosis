# CWRU Fault Diagnosis

面向变负载工况的轴承故障深度时序建模与多尺度注意力增强方法分析。

本项目基于 CWRU Bearing Data Center 轴承振动数据，围绕“同负载分类准确率高，是否代表跨负载工况也可靠”这一问题，构建防数据泄露的数据处理流程，并对比 CNN1D、ResNet1D 和 SE-ResNet1D 在同负载与跨负载场景下的表现。

## Current Progress

- Day 1：整理 CWRU 40 个 `.mat` 文件，建立 `metadata.csv`，检查 DE time series 是否可读取。
- Day 2：实现防数据泄露的数据处理流程，完成采样率统一、chronological split、滑窗和标准化。
- Day 3：实现 CNN1D baseline，完成 30 epoch 训练闭环。
- Day 4：实现 ResNet1D baseline，并完成训练曲线与日志保存。
- Day 5：完成 CNN1D 同负载实验，生成 4 个负载下的混淆矩阵。
- Day 6：完成 CNN1D、ResNet1D、SE-ResNet1D 跨负载实验，生成 accuracy / macro-F1 矩阵图。
- Attention：实现 SE-ResNet1D，用 SE channel attention 分析注意力增强对跨负载泛化的影响。

## Project Structure

```text
cwru_fault_diagnosis/
├── metadata.csv
├── README.md
├── data/
│   ├── raw_mat/                    # 本地原始 CWRU .mat 文件，未上传 GitHub
│   └── processed/
│       ├── dataset_info.txt        # 全局窗口数据集信息
│       └── by_load/                # 本地按负载划分的 npz 数据，未上传 GitHub
├── src/
│   ├── check_mat_files.py          # 检查 .mat 文件和 DE signal key
│   ├── dataset.py                  # 读取数据、重采样、时序切分、滑窗
│   ├── make_npz.py                 # 生成全局 train/test npz 数据
│   ├── make_npz_by_load.py         # 生成按负载划分的数据集
│   ├── train_cnn1d.py              # CNN1D 全局 baseline 训练
│   ├── train_resnet1d.py           # ResNet1D 全局 baseline 训练
│   ├── train_same_load_cnn1d.py    # CNN1D 同负载实验
│   ├── train_cross_load_cnn1d.py   # CNN1D 跨负载实验
│   ├── train_cross_load_resnet1d.py
│   ├── train_cross_load_se_resnet1d.py
│   └── models/
│       ├── cnn1d.py
│       ├── resnet1d.py
│       └── se_resnet1d.py
└── results/
    ├── logs/                       # 数据检查、训练日志、实验 summary
    ├── figures/
    │   ├── cross_load/             # 跨负载 accuracy / macro-F1 矩阵图
    │   └── same_load_confusion_matrices/
    └── checkpoints/                # 本地模型权重，未上传 GitHub
```

## Dataset

使用 CWRU Bearing Data Center 数据：

- 10 个类别：Normal、IR007、IR014、IR021、B007、B014、B021、OR007@6、OR014@6、OR021@6
- 4 个负载：0 / 1 / 2 / 3 hp
- 每个文件提取 Drive End vibration signal

当前窗口参数：

```text
window_size: 1024
stride: 512
train_ratio: 0.7
target_sampling_rate: 12000
```

全局数据集规模：

```text
X_train shape: (6518, 1, 1024)
y_train shape: (6518,)
X_test shape : (2764, 1, 1024)
y_test shape : (2764,)
```

## Data Leakage Prevention

本项目没有采用“先滑窗再随机 train_test_split”的方式，而是对每条原始振动信号先做 chronological split：

```text
原始长信号 -> 统一采样率 -> 前 70% 训练段 / 后 30% 测试段 -> 分别滑窗
```

这样可以避免相邻窗口同时进入训练集和测试集，降低数据泄露风险。

## Models

- CNN1D：基础 1D 卷积网络，用作 baseline。
- ResNet1D：加入 residual block，增强深层时序特征提取能力。
- SE-ResNet1D：在 ResNet1D block 中加入 SE channel attention，用于分析注意力增强效果。

## Experiments

### Same-load Classification

CNN1D 在 4 个负载内部训练/测试，结果如下：

```text
Load 0: Accuracy 0.9985, Macro-F1 0.9986
Load 1: Accuracy 1.0000, Macro-F1 1.0000
Load 2: Accuracy 1.0000, Macro-F1 1.0000
Load 3: Accuracy 1.0000, Macro-F1 1.0000
```

结论：同负载条件下，CWRU 10 类故障分类对 CNN1D 来说已经较容易。

### Cross-load Generalization

跨负载实验使用 source load 训练，并在 target load 测试。非对角线平均结果：

```text
CNN1D        cross-load avg accuracy: 0.8589
ResNet1D    cross-load avg accuracy: 0.8687
SE-ResNet1D cross-load avg accuracy: 0.8691
```

当 target load 为 3 hp 时，跨负载泛化最困难：

```text
CNN1D        target Load 3 avg accuracy: 0.7361
ResNet1D    target Load 3 avg accuracy: 0.7737
SE-ResNet1D target Load 3 avg accuracy: 0.7855
```

结论：同负载高准确率不等于跨负载可靠；变负载工况会带来明显分布偏移。ResNet1D 和 SE-ResNet1D 相比 CNN1D 有一定提升，但提升幅度有限，说明仅靠加深网络或通道注意力不能完全解决跨工况泛化问题。

## Run

检查原始 `.mat` 文件：

```bash
python src/check_mat_files.py
```

构建全局窗口数据：

```bash
python src/make_npz.py
```

构建按负载划分的数据：

```bash
python src/make_npz_by_load.py
```

训练 CNN1D baseline：

```bash
python src/train_cnn1d.py
```

训练 ResNet1D baseline：

```bash
python src/train_resnet1d.py
```

运行同负载实验：

```bash
python src/train_same_load_cnn1d.py
```

运行跨负载实验：

```bash
python src/train_cross_load_cnn1d.py
python src/train_cross_load_resnet1d.py
python src/train_cross_load_se_resnet1d.py
```

## Result Files

主要结果保存在：

```text
results/logs/same_load_cnn1d_summary.csv
results/logs/cross_load_cnn1d_summary.csv
results/logs/cross_load_resnet1d_summary.csv
results/logs/cross_load_se_resnet1d_summary.csv
results/figures/cross_load/
results/figures/same_load_confusion_matrices/
```

原始 `.mat`、生成的 `.npz` 数据和模型 checkpoint 文件较大，仅保存在本地，不上传 GitHub。
