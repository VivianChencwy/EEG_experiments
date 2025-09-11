"""
对比原始实验和改进实验的结果
"""

import os
import re
from pathlib import Path

def extract_accuracy_from_log(log_file):
    """从日志文件中提取准确率"""
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # 查找Mean Accuracy
        accuracy_matches = re.findall(r'Mean Accuracy: ([\d.]+)', content)
        if accuracy_matches:
            return float(accuracy_matches[-1])  # 取最后一个
        
        # 备用模式
        accuracy_matches = re.findall(r'平均准确率: ([\d.]+)%', content)
        if accuracy_matches:
            return float(accuracy_matches[-1]) / 100
            
        return None
    except Exception as e:
        print(f"读取日志文件错误 {log_file}: {e}")
        return None

def find_latest_logs():
    """找到最新的日志文件"""
    log_dirs = [d for d in Path('.').iterdir() if d.is_dir() and d.name.startswith('log_')]
    
    if not log_dirs:
        print("没有找到日志目录")
        return None, None
    
    # 获取最新的日志目录
    latest_log_dir = max(log_dirs, key=lambda x: x.stat().st_mtime)
    
    # 查找原始实验日志
    original_logs = list(latest_log_dir.glob('*ShallowFBCSPNet*'))
    improved_logs = list(latest_log_dir.glob('*Improved*'))
    
    # 如果没有找到改进日志，在所有目录中查找
    if not improved_logs:
        for log_dir in log_dirs:
            improved_logs.extend(log_dir.glob('*Improved*'))
    
    original_log = original_logs[0] if original_logs else None
    improved_log = improved_logs[0] if improved_logs else None
    
    return original_log, improved_log

def compare_experiments():
    """对比实验结果"""
    print("=" * 60)
    print("📊 EEG分类实验结果对比")
    print("=" * 60)
    
    # 查找日志文件
    original_log, improved_log = find_latest_logs()
    
    if original_log:
        print(f"✓ 找到原始实验日志: {original_log.name}")
    else:
        print("⚠️  未找到原始实验日志")
    
    if improved_log:
        print(f"✓ 找到改进实验日志: {improved_log.name}")
    else:
        print("⚠️  未找到改进实验日志")
    
    print()
    
    # 提取结果
    baseline_accuracy = 0.713  # 已知的baseline
    original_accuracy = None
    improved_accuracy = None
    
    if original_log:
        original_accuracy = extract_accuracy_from_log(original_log)
    
    if improved_log:
        improved_accuracy = extract_accuracy_from_log(improved_log)
    
    # 显示对比结果
    print("📈 准确率对比:")
    print("-" * 40)
    
    print(f"Baseline (已知):        {baseline_accuracy:.1%}")
    
    if original_accuracy:
        print(f"原始实验:              {original_accuracy:.1%}")
        if original_accuracy > baseline_accuracy:
            diff = original_accuracy - baseline_accuracy
            print(f"                       (+{diff:.1%} vs baseline)")
    else:
        print(f"原始实验:              未运行或未找到")
    
    if improved_accuracy:
        print(f"改进实验:              {improved_accuracy:.1%}")
        
        # 与baseline对比
        baseline_diff = improved_accuracy - baseline_accuracy
        baseline_pct = baseline_diff / baseline_accuracy * 100
        print(f"                       (+{baseline_diff:.1%} vs baseline, +{baseline_pct:.1f}%)")
        
        # 与原始实验对比
        if original_accuracy:
            original_diff = improved_accuracy - original_accuracy
            original_pct = original_diff / original_accuracy * 100
            print(f"                       (+{original_diff:.1%} vs 原始, +{original_pct:.1f}%)")
    else:
        print(f"改进实验:              未运行或未找到")
    
    print()
    
    # 改进效果评估
    if improved_accuracy:
        print("🎯 改进效果评估:")
        print("-" * 40)
        
        if improved_accuracy >= 0.85:
            print("🏆 优秀! 超越预期目标 (>85%)")
        elif improved_accuracy >= 0.78:
            print("🎉 优秀! 达到预期目标 (78-85%)")
        elif improved_accuracy >= 0.75:
            print("✅ 良好! 接近预期目标")
        elif improved_accuracy > baseline_accuracy:
            print("📈 有改进，可进一步优化")
        else:
            print("⚠️  未达预期，需要调整策略")
        
        print()
        
        # 技术改进分析
        print("🔧 技术改进分析:")
        improvement = improved_accuracy - baseline_accuracy
        
        if improvement >= 0.07:  # >7%
            print("• ICA伪影去除: 显著提升信号质量")
            print("• ERP频率优化: 有效增强P300检测")
            print("• 平衡采样: 改善类别不均衡")
        elif improvement >= 0.03:  # 3-7%
            print("• 改进方法有效，建议:")
            print("  - 尝试更多预处理技术")
            print("  - 调整模型超参数")
            print("  - 增加数据增强")
        else:
            print("• 改进幅度有限，建议:")
            print("  - 检查数据质量")
            print("  - 尝试不同模型架构")
            print("  - 增加训练数据")
    
    # 下一步建议
    print("💡 下一步建议:")
    print("-" * 40)
    
    if improved_accuracy and improved_accuracy >= 0.78:
        print("• ✅ 改进成功! 可以扩展到全数据集")
        print("• 🔄 尝试Transformer或混合模型获得进一步提升")
        print("• 📊 进行更详细的错误分析和优化")
    else:
        print("• 🔄 运行改进实验: python run_improved_experiment.py")
        print("• 🛠️  尝试其他改进方法:")
        print("  - CSP空间滤波")
        print("  - 注意力机制模型")
        print("  - 集成学习方法")
    
    return {
        'baseline': baseline_accuracy,
        'original': original_accuracy,
        'improved': improved_accuracy
    }

if __name__ == "__main__":
    results = compare_experiments()
    
    print("\n" + "=" * 60)
    print("📋 结果总结")
    print("=" * 60)
    
    if results['improved']:
        improvement = results['improved'] - results['baseline']
        print(f"最终准确率: {results['improved']:.1%}")
        print(f"总体提升: +{improvement:.1%}")
        
        if improvement >= 0.05:
            print("🎉 改进效果显著!")
        elif improvement > 0:
            print("📈 有一定改进效果")
        else:
            print("⚠️  需要进一步优化")
    else:
        print("请先运行改进实验:")
        print("python run_improved_experiment.py")