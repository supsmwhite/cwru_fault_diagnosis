# CWRU Fault Diagnosis

面向变负载工况的轴承故障深度时序建模与 CNN-BiLSTM-Attention 增强方法分析。

本项目基于 CWRU Bearing Data Center 轴承振动数据，重点不是做普通的 CWRU 分类，而是围绕一个更接近工程部署的问题展开：

> 同负载条件下的高准确率，是否代表模型在未知负载工况下也可靠？

项目先建立防数据泄露的数据处理流程，再从 CNN1D baseline 出发，逐步分析同负载、单源跨负载、多源跨负载、小样本跨负载和模型结构改进。最终主线模型为 **CNN-BiLSTM-Attention**：在 1D-CNN 局部特征提取基础上加入 BiLSTM 时序建模与 temporal attention 关键片段聚合。

## Current Progress

- Day 1：整理 10 类 × 4 负载 = 40 个 CWRU `.mat` 文件，建立 `metadata.csv`，完成 DE time series 读取检查。
- Day 2：完成采样率统一、chronological split、滑窗、z-score 标准化，生成训练/测试窗口数据。
- Day 3：实现 CNN1D baseline，完成 30 epoch 训练闭环。
- Day 4：实现 ResNet1D 与 SE-ResNet1D，对比更复杂模型的收益。
- Day 5：完成同负载实验，验证标准 CWRU 分类任务已接近饱和。
- Day 6：完成单源跨负载实验，发现 Load 3 是最困难目标工况。
- Day 7：完成 leave-one-load-out 多源跨负载实验，验证多工况覆盖能显著改善泛化。
- Day 8：完成 Load 3 小样本、数据增强、多随机种子实验。
- Day 9：实现 CNN-BiLSTM-Attention，并通过消融实验验证 BiLSTM 和 temporal attention 的贡献。

## Project Structure

```text
cwru_fault_diagnosis/
├── metadata.csv
├── README.md
├── data/
│   ├── raw_mat/                         # 本地原始 CWRU .mat 文件，未上传 GitHub
│   └── processed/
│       ├── dataset_info.txt             # 全局窗口数据集信息
│       └── by_load/                     # 本地按负载划分的 npz 数据，未上传 GitHub
├── src/
│   ├── check_mat_files.py               # 检查 .mat 文件和 DE signal key
│   ├── dataset.py                       # 读取数据、重采样、时序切分、滑窗
│   ├── make_npz.py                      # 生成全局 train/test npz 数据
│   ├── make_npz_by_load.py              # 生成按负载划分的数据集
│   ├── train_cnn1d.py                   # CNN1D baseline
│   ├── train_resnet1d.py                # ResNet1D baseline
│   ├── train_same_load_cnn1d.py         # 同负载实验
│   ├── train_cross_load_cnn1d.py        # 单源跨负载 CNN1D
│   ├── train_cross_load_resnet1d.py     # 单源跨负载 ResNet1D
│   ├── train_cross_load_se_resnet1d.py  # 单源跨负载 SE-ResNet1D
│   ├── train_leave_one_load_out_cnn1d.py
│   ├── train_leave_one_load_out_se_resnet1d.py
│   ├── train_few_shot_load3_compare_strict.py
│   ├── train_few_shot_load3_compare_augmented_multiseed.py
│   ├── train_few_shot_load3_cnn_lstm_attention_multiseed.py
│   ├── train_few_shot_load3_cnn_lstm_ablation_multiseed.py
│   └── models/
│       ├── cnn1d.py
│       ├── resnet1d.py
│       ├── se_resnet1d.py
│       └── cnn_lstm_attention.py
└── results/
    ├── logs/                            # 实验日志和 summary
    ├── figures/
    │   ├── cross_load/
    │   ├── leave_one_load_out/
    │   ├── few_shot_load3_augmented_multiseed/
    │   ├── few_shot_load3_cnn_lstm_attention_multiseed/
    │   ├── few_shot_load3_cnn_lstm_ablation_multiseed/
    │   └── same_load_confusion_matrices/
    └── checkpoints/                     # 本地模型权重，未上传 GitHub
```

## Dataset

使用 CWRU Bearing Data Center 数据：

- 10 个类别：Normal、IR007、IR014、IR021、B007、B014、B021、OR007@6、OR014@6、OR021@6
- 4 个负载：0 / 1 / 2 / 3 hp
- 每个文件提取 Drive End vibration signal

窗口参数：

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

## Leakage Prevention

本项目没有采用“先滑窗再随机 train_test_split”的方式，而是对每条原始振动信号先做 chronological split：

```text
原始长信号 -> 统一采样率 -> 前 70% 训练段 / 后 30% 测试段 -> 分别滑窗
```

在跨负载、小样本和增强实验中，当 Load 3 作为目标负载时，它不参与训练，也不参与 best epoch 选择，只作为最终测试集。

## Models

- **CNN1D**：基础 1D 卷积网络，作为强 baseline。
- **ResNet1D**：残差结构，用于检验加深网络是否能改善跨负载泛化。
- **SE-ResNet1D**：在 ResNet1D 中加入 SE channel attention。
- **CNN-BiLSTM**：CNN 提取局部特征，BiLSTM 建模窗口内部长程时序依赖。
- **CNN-BiLSTM-Attention**：在 CNN-BiLSTM 基础上加入 temporal attention，对关键时序片段进行加权聚合，是当前主线改进模型。

## Experiments And Findings

### 1. Same-load Classification

CNN1D 在 4 个负载内部训练/测试，结果接近饱和：

```text
Load 0: Accuracy 0.9985, Macro-F1 0.9986
Load 1: Accuracy 1.0000, Macro-F1 1.0000
Load 2: Accuracy 1.0000, Macro-F1 1.0000
Load 3: Accuracy 1.0000, Macro-F1 1.0000
```

结论：标准同负载 CWRU 分类相对容易，CNN1D 已经是很强的 baseline，因此项目重点转向跨负载和小样本泛化。

### 2. Single-source Cross-load

单源跨负载实验使用一个 source load 训练，并在另一个 target load 测试。非对角线平均准确率：

```text
CNN1D        : 0.8589
ResNet1D    : 0.8687
SE-ResNet1D : 0.8691
```

当 target load 为 3 hp 时，泛化最困难：

```text
CNN1D        target Load 3 avg accuracy: 0.7361
ResNet1D    target Load 3 avg accuracy: 0.7737
SE-ResNet1D target Load 3 avg accuracy: 0.7855
```

结论：同负载高准确率不等于跨负载可靠。负载变化带来明显分布偏移，单纯加深网络或加入 SE 注意力只能带来有限提升。

### 3. Multi-source Cross-load / Leave-one-load-out

多源跨负载实验使用 3 个负载训练，剩下 1 个负载作为未知目标工况：

```text
Train 1+2+3 -> Test 0
Train 0+2+3 -> Test 1
Train 0+1+3 -> Test 2
Train 0+1+2 -> Test 3
```

平均准确率：

```text
CNN1D        : 0.9710
SE-ResNet1D : 0.9716
```

目标 Load 3 仍然最困难，但相比单源跨负载明显提升：

```text
CNN1D        Load 3 accuracy: 0.8902, Macro-F1: 0.8524
SE-ResNet1D Load 3 accuracy: 0.9030, Macro-F1: 0.8970
```

结论：多源负载训练显著提升未知负载泛化能力，说明训练工况覆盖不足是跨负载性能下降的重要原因。

### 4. Few-shot Load 3 With Augmentation

困难场景固定为：

```text
Train: Load 0 + Load 1 + Load 2 的部分训练样本
Validation: Load 0 + Load 1 + Load 2 的 test_windows
Final Test: Load 3 test_windows
```

增强策略只作用于训练集：

```text
random amplitude scaling: 0.8 - 1.2
random Gaussian noise: SNR 10 - 30 dB
random time shift: up to 64 points
```

3-seed 统计结果显示，CNN1D 与 SE-ResNet1D 在小样本 Load 3 上仍存在较大波动：

```text
CNN1D 50%        : Acc 0.8745 ± 0.0216, F1 0.8427 ± 0.0230
SE-ResNet1D 50%  : Acc 0.9073 ± 0.0228, F1 0.8967 ± 0.0289
SE-ResNet1D 100% : Acc 0.8787 ± 0.0422, F1 0.8606 ± 0.0449
```

结论：增强和 SE 注意力有一定帮助，但仍不能稳定解决困难目标负载下的小样本泛化问题。

### 5. CNN-BiLSTM-Attention Improvement

为增强窗口内部时序建模能力，引入 CNN-BiLSTM-Attention：

```text
1D-CNN local feature extractor
-> BiLSTM temporal dependency modeling
-> temporal attention pooling
-> classifier
```

3-seed 小样本 Load 3 结果：

```text
CNN1D 10%                  : Acc 0.7603 ± 0.0224, F1 0.7199 ± 0.0278
CNN-BiLSTM-Attention 10%   : Acc 0.7884 ± 0.0999, F1 0.7671 ± 0.1171

CNN1D 30%                  : Acc 0.8626 ± 0.0135, F1 0.8338 ± 0.0177
CNN-BiLSTM-Attention 30%   : Acc 0.9239 ± 0.0375, F1 0.9207 ± 0.0408

CNN1D 50%                  : Acc 0.8773 ± 0.0136, F1 0.8481 ± 0.0108
CNN-BiLSTM-Attention 50%   : Acc 0.9377 ± 0.0791, F1 0.9359 ± 0.0821

CNN1D 100%                 : Acc 0.8659 ± 0.0556, F1 0.8231 ± 0.0748
CNN-BiLSTM-Attention 100%  : Acc 0.9710 ± 0.0281, F1 0.9710 ± 0.0280
```

结论：在强 CNN baseline 上加入 BiLSTM 和 temporal attention 后，30%、50%、100% 训练比例下均取得明显提升，说明长程时序依赖和关键时间片段聚合对未知负载 Load 3 识别有效。

### 6. Ablation Study

消融实验比较：

```text
CNN1D
CNN-BiLSTM
CNN-BiLSTM-Attention
```

3-seed 结果：

```text
10%:
CNN1D                  Acc 0.7603, F1 0.7199
CNN-BiLSTM             Acc 0.8379, F1 0.8092
CNN-BiLSTM-Attention   Acc 0.7884, F1 0.7671

30%:
CNN1D                  Acc 0.8626, F1 0.8338
CNN-BiLSTM             Acc 0.8583, F1 0.8360
CNN-BiLSTM-Attention   Acc 0.9239, F1 0.9207

50%:
CNN1D                  Acc 0.8773, F1 0.8481
CNN-BiLSTM             Acc 0.9058, F1 0.8954
CNN-BiLSTM-Attention   Acc 0.9377, F1 0.9359

100%:
CNN1D                  Acc 0.8659, F1 0.8231
CNN-BiLSTM             Acc 0.9253, F1 0.9161
CNN-BiLSTM-Attention   Acc 0.9710, F1 0.9710
```

结论：

- BiLSTM 在 10%、50%、100% 条件下明显优于 CNN1D，说明时序依赖建模是主要有效改进。
- Temporal attention 在 30%、50%、100% 条件下进一步提升性能，说明关键时间片段聚合具有价值。
- 在 10% 极小样本条件下，attention 不如单纯 CNN-BiLSTM，说明注意力权重学习需要一定样本覆盖，极小样本下可能不稳定。

## Overall Conclusions

1. CNN1D 是非常强的 baseline：同负载几乎满分，多源跨负载也能达到较高准确率。
2. 真正困难的问题不是普通故障分类，而是未知负载、小样本和分布偏移下的泛化。
3. ResNet1D 和 SE-ResNet1D 整体提升有限，说明单纯加深网络或通道重标定不是核心突破口。
4. 多源负载训练显著改善泛化，说明工况覆盖是关键因素。
5. CNN-BiLSTM-Attention 在困难 Load 3 小样本跨负载任务中显著优于 CNN1D，成为当前最有价值的模型改进。
6. 消融实验表明：BiLSTM 提供长程时序建模收益，attention 在样本较充分时进一步提升，但极小样本下可能不稳定。

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

运行同负载实验：

```bash
python src/train_same_load_cnn1d.py
```

运行单源跨负载实验：

```bash
python src/train_cross_load_cnn1d.py
python src/train_cross_load_resnet1d.py
python src/train_cross_load_se_resnet1d.py
```

运行多源跨负载实验：

```bash
python src/train_leave_one_load_out_cnn1d.py
python src/train_leave_one_load_out_se_resnet1d.py
```

运行 CNN-BiLSTM-Attention 小样本实验：

```bash
python src/train_few_shot_load3_cnn_lstm_attention_multiseed.py
```

运行消融实验：

```bash
python src/train_few_shot_load3_cnn_lstm_ablation_multiseed.py
```

## Result Files

主要结果保存在：

```text
results/logs/same_load_cnn1d_summary.csv
results/logs/cross_load_cnn1d_summary.csv
results/logs/cross_load_resnet1d_summary.csv
results/logs/cross_load_se_resnet1d_summary.csv
results/logs/leave_one_load_out_cnn1d_summary.csv
results/logs/leave_one_load_out_se_resnet1d_summary.csv
results/logs/few_shot_load3_cnn_lstm_attention_multiseed_mean_std.csv
results/logs/few_shot_load3_cnn_lstm_ablation_multiseed_mean_std.csv
results/figures/cross_load/
results/figures/leave_one_load_out/
results/figures/few_shot_load3_cnn_lstm_attention_multiseed/
results/figures/few_shot_load3_cnn_lstm_ablation_multiseed/
results/figures/same_load_confusion_matrices/
```

原始 `.mat`、生成的 `.npz` 数据和模型 checkpoint 文件较大，仅保存在本地，不上传 GitHub。
