#!/usr/bin/env python3
"""
测试训练进度打印功能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import create_model, train_model
from config import *
import numpy as np

def create_dummy_data():
    """创建虚拟数据用于测试"""
    # 生成模拟EEG数据
    n_samples = 320  # 较小的数据集用于快速测试
    n_channels = 32
    n_timepoints = 128
    
    # 创建带有一些模式的虚拟数据
    X = torch.randn(n_samples, n_channels, n_timepoints)
    
    # 创建有意义的标签 (oddball vs standard)
    y = torch.randint(0, 2, (n_samples,))
    
    # 添加一些模式使分类任务不那么随机
    for i in range(n_samples):
        if y[i] == 1:  # oddball事件
            # 在某些通道和时间点添加特征
            X[i, :8, 30:50] += 0.5  # P300相关通道的早期响应
            X[i, 15:25, 60:90] += 0.3  # 后期响应
    
    return X, y

def test_training_with_progress():
    """测试带有进度打印的训练过程"""
    
    print("创建虚拟EEG数据...")
    X, y = create_dummy_data()
    
    # 分割数据
    n_total = len(X)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    print(f"数据分割: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 创建模型
    print("创建EEGConformer模型...")
    model = create_model(n_channels=32, model_name='EEGConformer')
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"使用设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型（限制为少数几个epoch用于测试）
    print("\n开始训练测试...")
    
    # 临时修改配置用于快速测试
    original_max_epochs = MAX_EPOCHS
    original_patience = EARLY_STOPPING_PATIENCE
    
    # 修改全局配置（这里用monkey patching）
    import config
    config.MAX_EPOCHS = 5  # 只训练5个epoch
    config.EARLY_STOPPING_PATIENCE = 3
    
    try:
        final_accuracy = train_model(
            model, train_loader, val_loader, test_loader, 
            device, is_lda=False, max_epochs=5
        )
        
        print(f"\n最终测试准确率: {100 * final_accuracy:.2f}%")
        
    finally:
        # 恢复原始配置
        config.MAX_EPOCHS = original_max_epochs
        config.EARLY_STOPPING_PATIENCE = original_patience

if __name__ == "__main__":
    print("测试EEGConformer训练进度显示")
    print("="*50)
    test_training_with_progress()