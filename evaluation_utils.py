"""
Enhanced evaluation utilities for multi-dataset fusion experiments
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None
import pandas as pd
from collections import defaultdict
import logging

from config import (
    ENABLE_COMPREHENSIVE_EVALUATION, ENABLE_DOMAIN_ANALYSIS, ENABLE_SMALL_SAMPLE_ANALYSIS,
    SMALL_SAMPLE_SIZES, SMALL_SAMPLE_SUBJECTS
)


class ComprehensiveEvaluator:
    """全面评估器类"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.results_cache = {}

    def evaluate_model(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
                      device: torch.device, domain_info: Optional[Dict] = None) -> Dict:
        """
        全面评估模型性能

        Args:
            model: 待评估模型
            test_loader: 测试数据加载器
            device: 计算设备
            domain_info: 域信息（用于域间分析）

        Returns:
            评估结果字典
        """
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_domains = []

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:  # (data, labels, domain_info)
                    data, labels, domains = batch
                    all_domains.extend(domains.cpu().numpy() if torch.is_tensor(domains) else domains)
                else:
                    data, labels = batch
                    domains = None

                data, labels = data.to(device), labels.to(device)

                # 前向传播
                outputs = model(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 取第一个输出（预测结果）

                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # 转换为numpy数组
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)

        # 计算基础指标
        results = self._calculate_basic_metrics(predictions, labels, probabilities)

        # 域间分析
        if ENABLE_DOMAIN_ANALYSIS and len(all_domains) > 0 and domain_info:
            domain_results = self._analyze_domain_performance(
                predictions, labels, probabilities, all_domains, domain_info
            )
            results['domain_analysis'] = domain_results

        # 混淆矩阵分析
        results['confusion_matrix'] = self._analyze_confusion_matrix(predictions, labels)

        # 类别平衡分析
        results['class_balance'] = self._analyze_class_balance(predictions, labels)

        return results

    def _calculate_basic_metrics(self, predictions: np.ndarray, labels: np.ndarray,
                               probabilities: np.ndarray) -> Dict:
        """计算基础评估指标"""
        n_classes = len(np.unique(labels))

        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision_macro': precision_score(labels, predictions, average='macro', zero_division=0),
            'precision_micro': precision_score(labels, predictions, average='micro', zero_division=0),
            'recall_macro': recall_score(labels, predictions, average='macro', zero_division=0),
            'recall_micro': recall_score(labels, predictions, average='micro', zero_division=0),
            'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
            'f1_micro': f1_score(labels, predictions, average='micro', zero_division=0),
        }

        # ROC-AUC（仅适用于二分类或多分类概率）
        try:
            if n_classes == 2:
                metrics['roc_auc'] = roc_auc_score(labels, probabilities[:, 1])
            else:
                metrics['roc_auc_ovr'] = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(labels, probabilities, multi_class='ovo', average='macro')
        except Exception as e:
            self.logger.warning(f"Failed to calculate ROC-AUC: {e}")
            metrics['roc_auc'] = 0.0

        # 每个类别的详细指标
        per_class_precision = precision_score(labels, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(labels, predictions, average=None, zero_division=0)
        per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)

        metrics['per_class'] = {
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1': per_class_f1.tolist()
        }

        return metrics

    def _analyze_domain_performance(self, predictions: np.ndarray, labels: np.ndarray,
                                  probabilities: np.ndarray, domains: List,
                                  domain_info: Dict) -> Dict:
        """分析域间性能"""
        domain_results = {}
        unique_domains = np.unique(domains)

        for domain in unique_domains:
            domain_mask = np.array(domains) == domain
            domain_pred = predictions[domain_mask]
            domain_labels = labels[domain_mask]
            domain_probs = probabilities[domain_mask]

            if len(domain_pred) > 0:
                domain_metrics = self._calculate_basic_metrics(domain_pred, domain_labels, domain_probs)
                domain_results[str(domain)] = domain_metrics

        # 计算域间差异
        if len(unique_domains) > 1:
            domain_accuracies = [domain_results[str(d)]['accuracy'] for d in unique_domains
                               if str(d) in domain_results]
            domain_results['cross_domain_variance'] = np.var(domain_accuracies)
            domain_results['cross_domain_std'] = np.std(domain_accuracies)
            domain_results['min_domain_accuracy'] = np.min(domain_accuracies)
            domain_results['max_domain_accuracy'] = np.max(domain_accuracies)

        return domain_results

    def _analyze_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """分析混淆矩阵"""
        cm = confusion_matrix(labels, predictions)
        n_classes = cm.shape[0]

        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 计算对角线准确率
        diagonal_accuracy = np.diag(cm_normalized)

        results = {
            'matrix': cm.tolist(),
            'normalized_matrix': cm_normalized.tolist(),
            'per_class_accuracy': diagonal_accuracy.tolist(),
            'off_diagonal_errors': (cm - np.diag(np.diag(cm))).sum()
        }

        return results

    def _analyze_class_balance(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """分析类别平衡性"""
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)

        # 确保两个数组有相同的类别
        all_classes = np.union1d(unique_labels, unique_preds)
        label_dist = np.zeros(len(all_classes))
        pred_dist = np.zeros(len(all_classes))

        for i, cls in enumerate(all_classes):
            if cls in unique_labels:
                label_dist[i] = label_counts[unique_labels == cls][0]
            if cls in unique_preds:
                pred_dist[i] = pred_counts[unique_preds == cls][0]

        # 计算分布差异
        total_samples = len(labels)
        label_dist_norm = label_dist / total_samples
        pred_dist_norm = pred_dist / total_samples

        # KL散度
        kl_divergence = np.sum(label_dist_norm * np.log(
            (label_dist_norm + 1e-10) / (pred_dist_norm + 1e-10)
        ))

        results = {
            'true_distribution': label_dist_norm.tolist(),
            'predicted_distribution': pred_dist_norm.tolist(),
            'kl_divergence': float(kl_divergence),
            'class_balance_ratio': float(np.min(label_counts) / np.max(label_counts))
        }

        return results


class SmallSampleAnalyzer:
    """小样本分析器"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def analyze_sample_efficiency(self, model_factory, train_data: Dict, test_data: Dict,
                                device: torch.device) -> Dict:
        """
        分析模型的样本效率

        Args:
            model_factory: 模型工厂函数
            train_data: 训练数据字典
            test_data: 测试数据字典
            device: 计算设备

        Returns:
            样本效率分析结果
        """
        if not ENABLE_SMALL_SAMPLE_ANALYSIS:
            return {}

        results = {}

        # 分析不同试验数量的影响
        for n_trials in SMALL_SAMPLE_SIZES:
            trial_results = self._evaluate_with_limited_trials(
                model_factory, train_data, test_data, n_trials, device
            )
            results[f'trials_{n_trials}'] = trial_results

        # 分析不同被试数量的影响
        for n_subjects in SMALL_SAMPLE_SUBJECTS:
            subject_results = self._evaluate_with_limited_subjects(
                model_factory, train_data, test_data, n_subjects, device
            )
            results[f'subjects_{n_subjects}'] = subject_results

        return results

    def _evaluate_with_limited_trials(self, model_factory, train_data: Dict, test_data: Dict,
                                    n_trials: int, device: torch.device) -> Dict:
        """评估有限试验数量的效果"""
        # 实现有限试验数量的评估逻辑
        # 这里简化实现，实际使用时需要根据具体数据结构调整
        limited_train_data = self._limit_trials_per_subject(train_data, n_trials)

        # 创建并训练模型
        model = model_factory()
        model.to(device)

        # 训练模型（简化版本）
        # 实际实现需要完整的训练循环
        trained_model = self._quick_train(model, limited_train_data, device)

        # 评估
        evaluator = ComprehensiveEvaluator(self.logger)
        results = evaluator.evaluate_model(trained_model, test_data, device)

        return results

    def _evaluate_with_limited_subjects(self, model_factory, train_data: Dict, test_data: Dict,
                                      n_subjects: int, device: torch.device) -> Dict:
        """评估有限被试数量的效果"""
        limited_train_data = self._limit_subjects(train_data, n_subjects)

        model = model_factory()
        model.to(device)

        trained_model = self._quick_train(model, limited_train_data, device)

        evaluator = ComprehensiveEvaluator(self.logger)
        results = evaluator.evaluate_model(trained_model, test_data, device)

        return results

    def _limit_trials_per_subject(self, data: Dict, n_trials: int) -> Dict:
        """限制每个被试的试验数量"""
        # 实现试验数量限制逻辑
        # 这里需要根据具体的数据结构来实现
        return data  # 简化返回

    def _limit_subjects(self, data: Dict, n_subjects: int) -> Dict:
        """限制被试数量"""
        # 实现被试数量限制逻辑
        return data  # 简化返回

    def _quick_train(self, model: torch.nn.Module, train_data: Dict, device: torch.device) -> torch.nn.Module:
        """快速训练模型（简化版本）"""
        # 这里需要实现简化的训练逻辑
        # 实际使用时应该调用完整的训练流程
        return model


class CrossDatasetEvaluator:
    """跨数据集评估器"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def evaluate_generalization(self, model: torch.nn.Module, datasets: Dict[str, torch.utils.data.DataLoader],
                               device: torch.device) -> Dict:
        """
        评估跨数据集泛化能力

        Args:
            model: 训练好的模型
            datasets: 各数据集的测试加载器
            device: 计算设备

        Returns:
            泛化能力评估结果
        """
        results = {}
        evaluator = ComprehensiveEvaluator(self.logger)

        for dataset_name, test_loader in datasets.items():
            self.logger.info(f"Evaluating on {dataset_name}")
            dataset_results = evaluator.evaluate_model(model, test_loader, device)
            results[dataset_name] = dataset_results

        # 计算跨数据集统计
        accuracies = [results[name]['accuracy'] for name in datasets.keys()]
        results['cross_dataset_stats'] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'accuracy_range': np.max(accuracies) - np.min(accuracies)
        }

        return results


class ResultsComparator:
    """结果比较器"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def compare_methods(self, results: Dict[str, Dict]) -> Dict:
        """
        比较不同方法的结果

        Args:
            results: {method_name: evaluation_results}

        Returns:
            比较结果
        """
        comparison = {}

        # 提取关键指标
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        method_names = list(results.keys())

        for metric in metrics:
            metric_values = {}
            for method in method_names:
                if metric in results[method]:
                    metric_values[method] = results[method][metric]

            if metric_values:
                comparison[metric] = {
                    'values': metric_values,
                    'best_method': max(metric_values.keys(), key=lambda k: metric_values[k]),
                    'worst_method': min(metric_values.keys(), key=lambda k: metric_values[k]),
                    'improvement': self._calculate_improvement(metric_values)
                }

        # 统计显著性检验（如果需要）
        comparison['statistical_tests'] = self._perform_statistical_tests(results)

        return comparison

    def _calculate_improvement(self, metric_values: Dict[str, float]) -> Dict:
        """计算改进程度"""
        if 'baseline' in metric_values:
            baseline_value = metric_values['baseline']
            improvements = {}
            for method, value in metric_values.items():
                if method != 'baseline':
                    improvement = (value - baseline_value) / baseline_value * 100
                    improvements[method] = improvement
            return improvements
        return {}

    def _perform_statistical_tests(self, results: Dict[str, Dict]) -> Dict:
        """执行统计显著性检验"""
        # 这里可以实现t检验、Wilcoxon符号秩检验等
        # 简化实现
        return {"note": "Statistical tests not implemented in this version"}


class VisualizationUtils:
    """可视化工具类"""

    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str] = None,
                            save_path: str = None) -> plt.Figure:
        """绘制混淆矩阵"""
        fig, ax = plt.subplots(figsize=(8, 6))

        if SEABORN_AVAILABLE:
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        else:
            im = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            if class_names is not None:
                ax.set_xticks(np.arange(len(class_names)))
                ax.set_yticks(np.arange(len(class_names)))
                ax.set_xticklabels(class_names)
                ax.set_yticklabels(class_names)

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_method_comparison(comparison_results: Dict, metric: str = 'accuracy',
                             save_path: str = None) -> plt.Figure:
        """绘制方法比较图"""
        if metric not in comparison_results:
            raise ValueError(f"Metric {metric} not found in comparison results")

        values = comparison_results[metric]['values']
        methods = list(values.keys())
        scores = list(values.values())

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(methods, scores, color=['skyblue' if m != 'baseline' else 'lightcoral' for m in methods])
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison Across Methods')

        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_sample_efficiency(sample_analysis: Dict, save_path: str = None) -> plt.Figure:
        """绘制样本效率图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 试验数量效果
        trial_keys = [k for k in sample_analysis.keys() if k.startswith('trials_')]
        if trial_keys:
            trial_nums = [int(k.split('_')[1]) for k in trial_keys]
            trial_accs = [sample_analysis[k]['accuracy'] for k in trial_keys]

            ax1.plot(trial_nums, trial_accs, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Trials per Subject')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Effect of Trial Number on Performance')
            ax1.grid(True, alpha=0.3)

        # 被试数量效果
        subject_keys = [k for k in sample_analysis.keys() if k.startswith('subjects_')]
        if subject_keys:
            subject_nums = [int(k.split('_')[1]) for k in subject_keys]
            subject_accs = [sample_analysis[k]['accuracy'] for k in subject_keys]

            ax2.plot(subject_nums, subject_accs, 's-', linewidth=2, markersize=8, color='orange')
            ax2.set_xlabel('Number of Subjects')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Effect of Subject Number on Performance')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig