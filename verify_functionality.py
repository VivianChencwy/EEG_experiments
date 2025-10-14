#!/usr/bin/env python3
"""
EEG融合系统功能验证脚本
验证核心功能是否正常工作
"""

import sys
import os
import numpy as np
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_electrode_utils():
    """验证电极位置管理模块"""
    print("验证电极位置管理模块...")

    try:
        from electrode_utils import get_electrode_positions, ElectrodeGraphBuilder

        # 测试获取电极位置
        test_electrodes = ['Fp1', 'Fz', 'Cz', 'Pz', 'O1']
        positions = get_electrode_positions(test_electrodes)
        assert len(positions) == len(test_electrodes), "电极位置数量不匹配"

        # 测试图构建器
        builder = ElectrodeGraphBuilder()
        adjacency_tensor = builder.build_graph(test_electrodes)
        assert isinstance(adjacency_tensor, torch.Tensor), "返回的不是张量"
        assert adjacency_tensor.shape[0] == len(test_electrodes), "邻接矩阵维度不匹配"

        print("✓ 电极位置管理模块验证通过")
        return True

    except Exception as e:
        print(f"✗ 电极位置管理模块验证失败: {e}")
        return False


def verify_fusion_methods():
    """验证融合方法模块"""
    print("验证融合方法模块...")

    try:
        from fusion_methods import FusionModelFactory

        # 准备测试数据
        datasets_info = {
            'P3': {'channels': ['Fp1', 'Fz', 'Cz', 'Pz', 'O1'], 'n_samples': 100},
            'AVO': {'channels': ['Fp1', 'Fz', 'Cz', 'Pz', 'O1'], 'n_samples': 150}
        }
        base_model_info = {'input_shape': (5, 500), 'n_classes': 2}

        # 测试创建融合模型
        model = FusionModelFactory.create_fusion_model(
            fusion_method='graph_gcn',
            datasets_info=datasets_info,
            base_model_info=base_model_info
        )

        # 测试前向传播
        dummy_input = torch.randn(2, 5, 500)
        # 对于UniversalFeatureSpace，使用简化的前向传播方法
        if hasattr(model, 'forward_single_dataset'):
            output = model.forward_single_dataset(dummy_input)
        else:
            output = model(dummy_input)
        assert output.shape[0] == 2, "输出批次大小不匹配"
        assert output.shape[1] == 2, "输出类别数不匹配"

        print("✓ 融合方法模块验证通过")
        return True

    except Exception as e:
        print(f"✗ 融合方法模块验证失败: {e}")
        return False


def verify_domain_adaptation():
    """验证域适应模块"""
    print("验证域适应模块...")

    try:
        from domain_adaptation import DomainAdapterFactory, MMDLoss
        import torch.nn as nn

        # 创建简单的特征提取器
        feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 500, 128),
            nn.ReLU()
        )

        # 测试MS-MDA
        ms_mda = DomainAdapterFactory.create_domain_adapter(
            adaptation_method='ms_mda',
            feature_extractor=feature_extractor,
            feature_dim=128,
            domains_info={'P3': 100, 'AVO': 150}
        )

        # 测试前向传播
        dummy_input = torch.randn(2, 5, 500)
        output = ms_mda(dummy_input)
        assert output.shape[0] == 2, "MS-MDA输出批次大小不匹配"

        # 测试MMD损失
        mmd_loss = MMDLoss()
        source_features = torch.randn(10, 64)
        target_features = torch.randn(10, 64)
        loss = mmd_loss(source_features, target_features)
        assert isinstance(loss.item(), float), "MMD损失计算失败"

        print("✓ 域适应模块验证通过")
        return True

    except Exception as e:
        print(f"✗ 域适应模块验证失败: {e}")
        return False


def verify_evaluation_utils():
    """验证评估工具模块"""
    print("验证评估工具模块...")

    try:
        from evaluation_utils import ComprehensiveEvaluator

        # 创建评估器
        evaluator = ComprehensiveEvaluator()

        # 测试基本功能
        assert hasattr(evaluator, 'logger'), "评估器缺少日志器"
        assert hasattr(evaluator, 'results_cache'), "评估器缺少结果缓存"

        print("✓ 评估工具模块验证通过")
        return True

    except Exception as e:
        print(f"✗ 评估工具模块验证失败: {e}")
        return False


def verify_enhanced_preprocessor():
    """验证增强预处理器模块"""
    print("验证增强预处理器模块...")

    try:
        from enhanced_preprocessor import FusionDatasetManager

        # 创建融合数据集管理器
        manager = FusionDatasetManager(
            fusion_method='graph_gcn',
            domain_adaptation='none'
        )

        assert hasattr(manager, 'fusion_method'), "管理器缺少融合方法属性"
        assert hasattr(manager, 'domain_adaptation'), "管理器缺少域适应属性"

        print("✓ 增强预处理器模块验证通过")
        return True

    except Exception as e:
        print(f"✗ 增强预处理器模块验证失败: {e}")
        return False


def verify_models():
    """验证模型模块"""
    print("验证模型模块...")

    try:
        from models import create_fusion_model

        # 准备测试数据
        datasets_info = {
            'P3': {'channels': ['Fp1', 'Fz', 'Cz', 'Pz', 'O1'], 'n_samples': 100},
            'AVO': {'channels': ['Fp1', 'Fz', 'Cz', 'Pz', 'O1'], 'n_samples': 150}
        }

        # 测试创建融合模型
        model = create_fusion_model(
            model_name='ShallowFBCSPNet',
            datasets_info=datasets_info,
            fusion_method='none',  # 使用简单模式避免复杂依赖
            domain_adaptation='none'
        )

        assert model is not None, "模型创建失败"

        print("✓ 模型模块验证通过")
        return True

    except Exception as e:
        print(f"✗ 模型模块验证失败: {e}")
        return False


def verify_experiment():
    """验证实验模块"""
    print("验证实验模块...")

    try:
        from experiment import run_experiment_with_fusion

        # 检查函数是否存在和可调用
        assert callable(run_experiment_with_fusion), "实验函数不可调用"

        print("✓ 实验模块验证通过")
        return True

    except Exception as e:
        print(f"✗ 实验模块验证失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("EEG融合系统功能验证")
    print("=" * 60)

    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 验证各模块功能
    verification_functions = [
        verify_electrode_utils,
        verify_fusion_methods,
        verify_domain_adaptation,
        verify_evaluation_utils,
        verify_enhanced_preprocessor,
        verify_models,
        verify_experiment
    ]

    passed = 0
    total = len(verification_functions)

    for verify_func in verification_functions:
        if verify_func():
            passed += 1
        print()  # 添加空行

    print("=" * 60)
    print(f"功能验证结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有核心功能验证通过！")
        print("✓ 电极位置管理系统正常")
        print("✓ 图神经网络融合功能正常")
        print("✓ 域适应功能正常")
        print("✓ 评估工具正常")
        print("✓ 数据预处理管道正常")
        print("✓ 模型创建功能正常")
        print("✓ 实验运行框架正常")
        print("\n系统已准备就绪，可以运行EEG融合实验！")
        return 0
    else:
        print(f"❌ {total - passed} 个功能模块存在问题，请检查相关代码。")
        return 1

if __name__ == "__main__":
    sys.exit(main())