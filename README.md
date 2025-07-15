# EEG Experiments - Refactored Code

这是一个重构后的EEG实验代码库，用于处理P3和Active Visual Oddball（AVO）数据集的分类任务。

## 文件结构

```
├── config.py           # 配置文件，包含所有实验参数
├── constants.py        # 常量定义，包含通道名称等
├── preprocessor.py     # 预处理器模块
├── models.py          # 模型定义和训练相关函数
├── utils.py           # 工具函数模块
├── experiment.py      # 实验逻辑模块
├── main.py            # 主程序入口
├── experiment_logger.py # 实验日志记录器（原有文件）
└── README.md          # 说明文档
```

## 主要改进

1. **模块化设计**：将单一的大文件拆分为多个功能明确的模块
2. **配置分离**：所有配置参数集中在`config.py`中，便于修改
3. **类型修复**：修复了原代码中的类型错误和linter问题
4. **文档完善**：每个函数都有详细的docstring说明
5. **代码复用**：提取了公共功能到工具函数中

## 使用方法

### 1. 配置实验参数

在`config.py`中修改实验配置：

```python
# 数据集选择
data_dir = P3_DATA_DIR  # 或 AVO_DATA_DIR
dataset = 'P3 Raw Data BIDS-Compatible'  # 或 'ds005863'
use_combined_datasets = False  # 是否使用联合数据集

# 实验参数
electrode_list = 'common'  # 或 'all'
classifier = 'ShallowFBCSPNet'  # 或 'lda'
separate_subject_classification = False  # 是否分别训练每个subject
```

### 2. 运行实验

```bash
python main.py
```

### 3. 实验配置选项

#### 数据集选项：
- **Option 1**: 仅使用P3数据集
- **Option 2**: 仅使用AVO数据集  
- **Option 3**: 使用联合数据集（P3+AVO）

#### 电极配置：
- `'common'`: 使用两个数据集的公共电极
- `'all'`: 使用数据集特定的所有电极

#### 分类器选择：
- `'ShallowFBCSPNet'`: 深度学习模型
- `'lda'`: 线性判别分析

#### 训练策略：
- `separate_subject_classification=True`: 为每个subject分别训练模型
- `separate_subject_classification=False`: 使用pooled data训练单个模型

## 代码结构说明

### config.py
- 所有实验配置参数
- 超参数设置
- 路径配置

### constants.py
- 电极通道名称定义
- 事件代码常量
- 数据处理常量

### preprocessor.py
- `OddballPreprocessor`类：用于EEG数据预处理
- 包含滤波、重采样、事件提取等功能

### models.py
- 模型创建函数
- 数据标准化函数
- 早停机制
- 模型训练和评估函数

### utils.py
- 数据加载工具函数
- 统计计算函数
- 数据分割和DataLoader创建
- 通用实验运行函数

### experiment.py
- 联合数据集实验逻辑
- 单数据集实验逻辑
- 分离subject实验逻辑

### main.py
- 主程序入口
- 实验配置验证
- 日志设置
- 实验流程控制

## 输出结果

实验运行后会生成：
- 控制台输出：实时进度和结果
- 日志文件：详细的实验记录（在`log/`目录）
- 统计结果：平均准确率、置信区间、最佳/最差subject

## 注意事项

1. 确保所有依赖包已安装
2. 检查数据路径配置是否正确
3. 联合数据集实验会自动使用公共电极
4. 实验结果与原始代码完全一致

## 扩展性

新的重构结构使得代码更容易扩展：
- 添加新的预处理器：在`preprocessor.py`中添加新类
- 添加新的模型：在`models.py`中添加新的模型创建函数
- 添加新的数据集：在相应模块中添加处理逻辑
- 修改实验流程：在`experiment.py`中添加新的实验函数 