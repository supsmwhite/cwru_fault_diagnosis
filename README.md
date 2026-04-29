# CWRU Fault Diagnosis

面向变负载工况的轴承故障深度时序建模与多尺度注意力增强方法分析。

本项目基于 CWRU Bearing Data Center 轴承振动数据，重点不是做普通的 CWRU 分类，而是围绕一个更接近工程部署的问题展开：

> 同负载条件下的高准确率，是否代表模型在未知负载工况下也可靠？

项目构建了防数据泄露的数据处理流程，并比较 CNN1D、ResNet1D、SE-ResNet1D 在同负载、单源跨负载、多源跨负载、小样本和数据增强条件下的表现。

## Current Progress

- Day 1：整理 10 类 × 4 负载 = 40 个 CWRU `.mat` 文件，建立 `metadata.csv`，完成 DE time series 读取检查。
- Day 2：完成采样率统一、chronological split、滑窗、z-score 标准化，生成训练/测试窗口数据。
- Day 3：实现 CNN1D baseline，完成 30 epoch 训练闭环。
- Day 4：实现 ResNet1D baseline。
- Day 5：完成 CNN1D 同负载实验，生成 4 个负载下的混淆矩阵。
- Day 6：完成 CNN1D、ResNet1D、SE-ResNet1D 单源跨负载实验。
- Day 7：完成 leave-one-load-out 多源跨负载实验。
- Day 8：完成 Load 3 小样本跨负载实验、严格评估版本、训练增强版本和 3-seed 统计版本。

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
│   ├── train_few_shot_load3_compare.py
│   ├── train_few_shot_load3_compare_strict.py
│   ├── train_few_shot_load3_compare_augmented.py
│   ├── train_few_shot_load3_compare_augmented_multiseed.py
│   └── models/
│       ├── cnn1d.py
│       ├── resnet1d.py
│       └── se_resnet1d.py
└── results/
    ├── logs/                            # 实验日志和 summary
    ├── figures/
    │   ├── cross_load/
    │   ├── leave_one_load_out/
    │   ├── few_shot_load3/
    │   ├── few_shot_load3_strict/
    │   ├── few_shot_load3_augmented/
    │   ├── few_shot_load3_augmented_multiseed/
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

## Data Leakage Prevention

本项目没有采用“先滑窗再随机 train_test_split”的方式，而是对每条原始振动信号先做 chronological split：

```text
原始长信号 -> 统一采样率 -> 前 70% 训练段 / 后 30% 测试段 -> 分别滑窗
```

这样可以避免相邻窗口同时进入训练集和测试集，降低数据泄露风险。跨负载、小样本和增强实验中，目标负载 Load 3 只作为最终测试集时，不参与训练和 best epoch 选择。

## Models

- CNN1D：基础 1D 卷积网络，用作强 baseline。
- ResNet1D：加入 residual block，增强深层时序特征提取能力。
- SE-ResNet1D：在 ResNet1D block 中加入 SE channel attention，用于分析通道注意力对跨负载泛化的影响。

## Experiments And Findings

### 1. Same-load Classification

CNN1D 在 4 个负载内部训练/测试，结果接近饱和：

```text
Load 0: Accuracy 0.9985, Macro-F1 0.9986
Load 1: Accuracy 1.0000, Macro-F1 1.0000
Load 2: Accuracy 1.0000, Macro-F1 1.0000
Load 3: Accuracy 1.0000, Macro-F1 1.0000
```

结论：标准同负载 CWRU 10 类故障分类相对容易，简单 CNN1D 已经足够强。因此本项目后续重点转向跨负载泛化，而不是继续在同负载准确率上追求微小提升。

### 2. Single-source Cross-load Generalization

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

结论：同负载高准确率不等于跨负载可靠。负载变化会带来明显分布偏移，单纯加深模型或加入 SE 注意力只能带来有限提升。

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

结论：多源负载训练显著提升未知负载泛化能力。这说明跨负载性能下降的重要原因不是模型完全学不会故障特征，而是训练工况覆盖不足。

### 4. Few-shot Load 3 Strict Evaluation

小样本实验固定：

```text
Train: Load 0 + Load 1 + Load 2 的部分训练样本
Validation: Load 0 + Load 1 + Load 2 的 test_windows
Final Test: Load 3 test_windows
```

严格版本中，Load 3 不参与训练，也不参与 best epoch 选择。

部分结果如下：

```text
CNN1D 10%: Acc 0.7803, Macro-F1 0.7364
CNN1D 30%: Acc 0.8745, Macro-F1 0.8479
CNN1D 50%: Acc 0.8274, Macro-F1 0.8031
CNN1D 100%: Acc 0.7974, Macro-F1 0.7478

SE-ResNet1D 10%: Acc 0.7646, Macro-F1 0.7219
SE-ResNet1D 30%: Acc 0.8802, Macro-F1 0.8743
SE-ResNet1D 50%: Acc 0.8417, Macro-F1 0.8038
SE-ResNet1D 100%: Acc 0.8545, Macro-F1 0.8395
```

结论：严格未知负载评估下，小样本性能并不随训练比例单调提升。这说明源负载验证集表现接近满分时，并不一定能可靠预测目标负载 Load 3 的泛化表现。

### 5. Few-shot With Training Augmentation

增强策略只作用于训练集：

```text
random amplitude scaling: 0.8 - 1.2
random Gaussian noise: SNR 10 - 30 dB
random time shift: up to 64 points
```

单 seed 增强实验中，SE-ResNet1D 在部分设置下明显改善 Load 3：

```text
SE-ResNet1D 30%: Acc 0.9358, Macro-F1 0.9331
SE-ResNet1D 100%: Acc 0.9615, Macro-F1 0.9613
```

结论：适度训练增强可以提升困难目标负载上的鲁棒性，尤其能帮助 SE-ResNet1D 在部分小样本比例下获得更好的 Macro-F1。

### 6. Multi-seed Augmented Few-shot

为了避免单次随机种子带来偶然结论，增强版小样本实验进一步使用 3 个随机种子：

```text
seeds: 42 / 2024 / 3407
```

均值 ± 标准差结果：

```text
CNN1D 10%: Acc 0.8312 ± 0.0550, F1 0.7982 ± 0.0669
CNN1D 30%: Acc 0.8745 ± 0.0162, F1 0.8399 ± 0.0176
CNN1D 50%: Acc 0.8745 ± 0.0216, F1 0.8427 ± 0.0230
CNN1D 100%: Acc 0.8497 ± 0.0466, F1 0.8089 ± 0.0572

SE-ResNet1D 10%: Acc 0.8493 ± 0.0363, F1 0.8201 ± 0.0348
SE-ResNet1D 30%: Acc 0.8835 ± 0.0691, F1 0.8642 ± 0.0862
SE-ResNet1D 50%: Acc 0.9073 ± 0.0228, F1 0.8967 ± 0.0289
SE-ResNet1D 100%: Acc 0.8787 ± 0.0422, F1 0.8606 ± 0.0449
```

结论：在增强和多随机种子统计下，SE-ResNet1D 在 50% 训练样本比例下表现最好，说明通道注意力和训练增强对困难跨负载场景有一定帮助。但结果仍有随机波动，不能简单宣称复杂模型全面碾压 CNN1D。

## Overall Conclusions

1. CNN1D 是非常强的 MVP baseline：在同负载任务中几乎满分，在多源跨负载中也能达到约 97% 平均准确率。
2. 跨负载泛化的主要瓶颈不是普通分类能力，而是负载变化造成的数据分布偏移。
3. 单源跨负载显著困难，多源负载训练能明显改善未知负载泛化。
4. ResNet1D 和 SE-ResNet1D 的整体提升有限，但 SE-ResNet1D 在困难 Load 3、小样本增强和 Macro-F1 指标上体现出一定价值。
5. 严格评估比直接在目标负载上选 best epoch 更可信，也更能反映真实部署场景。
6. 后续提升方向应优先考虑鲁棒训练、数据增强、多源工况覆盖和更稳定的跨工况特征学习，而不是单纯堆叠更复杂模型。

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

训练 baseline：

```bash
python src/train_cnn1d.py
python src/train_resnet1d.py
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

运行 Load 3 小样本实验：

```bash
python src/train_few_shot_load3_compare_strict.py
python src/train_few_shot_load3_compare_augmented.py
python src/train_few_shot_load3_compare_augmented_multiseed.py
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
results/logs/few_shot_load3_compare_strict_summary.csv
results/logs/few_shot_load3_compare_augmented_summary.csv
results/logs/few_shot_load3_augmented_multiseed_mean_std.csv
results/figures/cross_load/
results/figures/leave_one_load_out/
results/figures/few_shot_load3_augmented_multiseed/
results/figures/same_load_confusion_matrices/
```

原始 `.mat`、生成的 `.npz` 数据和模型 checkpoint 文件较大，仅保存在本地，不上传 GitHub。
