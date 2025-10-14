# TF-DWT优化计划 - AVO目标准确率提升至0.65+

## 当前状态
- 配置: P3=80 trials, AVO=10 trials (AVO是目标)
- 目标: AVO准确率 ≥ 0.65
- 基线: 需要建立（0930/中无AVO目标的TF-DWT结果）

## 优化策略

### 策略1: 增大域权重上限 (w_small cap)
**当前**: `w_small_target = min(w_small_target, 3.0)` (第320行)
**问题**: 8:1比例下，理论权重√8≈2.83，上限3.0限制了小域强调
**优化**:
- 版本1a: cap=5.0
- 版本1b: cap=6.0
- 版本1c: cap=8.0 (更激进)

### 策略2: 调整MMD对齐强度
**当前**: λ_mmd=0.3 (8:1比例) (第209行)
**问题**: 过强的对齐可能干扰小域特征学习
**优化**:
- 版本2a: λ_mmd=0.15 (减半)
- 版本2b: λ_mmd=0.1 (保守)
- 版本2c: 动态调整 - 初期0.3，后期0.1

### 策略3: Batch Size调整
**当前**: BATCH_SIZE=128
**问题**: AVO每个subject只有10 trials，batch太大导致训练不稳定
**优化**:
- 版本3a: BATCH_SIZE=64
- 版本3b: BATCH_SIZE=32
- 版本3c: BATCH_SIZE=16

### 策略4: Learning Rate优化
**当前**: LR=0.001 (默认)
**问题**: 小数据集可能需要更小的学习率避免过拟合
**优化**:
- 版本4a: LR=0.0005
- 版本4b: LR=0.0003
- 版本4c: 动态LR - 小域用更小LR

### 策略5: Warmup调整
**当前**: warmup = max(2, min(5, int(0.1 * MAX_EPOCHS))) = 5 epochs (第210行)
**问题**: 5 epochs可能太短，小域还没充分学习
**优化**:
- 版本5a: warmup=10 epochs
- 版本5b: warmup=20 epochs
- 版本5c: warmup=50 epochs (10% of 500)

### 策略6: Early Stopping放宽
**当前**: EARLY_STOPPING_PATIENCE (需检查config)
**问题**: 过早停止可能导致小域欠拟合
**优化**:
- 版本6a: patience += 50%
- 版本6b: patience += 100%
- 版本6c: 只在大域也下降时才停止

### 策略7: 防护机制调整
**当前**: 连续3次下降就降权 (第291, 300行)
**问题**: 太敏感，限制了探索
**优化**:
- 版本7a: 连续5次才降权
- 版本7b: 下降幅度从0.8改为0.9 (更温和)
- 版本7c: 完全禁用防护机制

### 策略8: 数据增强
**问题**: AVO样本太少
**优化**:
- 版本8a: 增加噪声增强强度
- 版本8b: 增加时间平移范围
- 版本8c: 启用mixup/cutmix

## 实验计划

### Phase 1: 单因素测试 (建立基线)
1. **Baseline**: 运行原始TF-DWT，记录AVO准确率
2. **Strategy 1**: 测试权重上限 (1a, 1b, 1c)
3. **Strategy 2**: 测试MMD强度 (2a, 2b, 2c)
4. **Strategy 3**: 测试Batch Size (3a, 3b, 3c)

### Phase 2: 组合优化
根据Phase 1结果，组合最优参数：
- 最优权重上限 + 最优MMD
- 最优batch size + 最优LR
- 最优warmup + 最优early stopping

### Phase 3: 精细调优
在Phase 2最优组合基础上微调，目标AVO ≥ 0.65

## 执行策略
- 每个版本运行完整的5-fold CV
- 记录AVO准确率、标准差、AUC
- 达到0.65目标后立即保存配置并停止
- 未达到则继续下一个策略

## 输出文件命名
`tfdwt_optimized_v{strategy}_{version}_detailed_{timestamp}.csv`
例如: `tfdwt_optimized_v1a_cap5_detailed_20251001.csv`
