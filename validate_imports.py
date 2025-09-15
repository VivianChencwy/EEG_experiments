#!/usr/bin/env python3
"""
简单的导入验证脚本：确保所有模块可以正确导入
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_import(module_name, description):
    """测试单个模块导入"""
    try:
        __import__(module_name)
        print(f"✓ {description}: 导入成功")
        return True
    except Exception as e:
        print(f"✗ {description}: 导入失败 - {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("EEG融合系统模块导入验证")
    print("=" * 50)

    modules_to_test = [
        ('config', '配置模块'),
        ('electrode_utils', '电极位置管理模块'),
        ('fusion_methods', '融合方法模块'),
        ('domain_adaptation', '域适应模块'),
        ('evaluation_utils', '评估工具模块'),
        ('enhanced_preprocessor', '增强预处理器模块'),
        ('experiment', '实验模块'),
        ('models', '模型模块')
    ]

    passed = 0
    total = len(modules_to_test)

    for module_name, description in modules_to_test:
        if test_import(module_name, description):
            passed += 1

    print("\n" + "=" * 50)
    print(f"导入测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有模块导入成功！系统已准备就绪。")
        return 0
    else:
        print(f"❌ {total - passed} 个模块导入失败。")
        return 1

if __name__ == "__main__":
    sys.exit(main())