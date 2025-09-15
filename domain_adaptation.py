"""
Domain adaptation methods for multi-dataset EEG fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math

from config import (
    MS_MDA_ADAPTATION_WEIGHT, MS_MDA_ENSEMBLE_METHOD, MS_MDA_HIDDEN_DIM, MS_MDA_TEMPERATURE,
    ADVERSARIAL_WEIGHT, DISCRIMINATOR_HIDDEN_DIM, DISCRIMINATOR_LEARNING_RATE,
    GRADIENT_REVERSAL_LAMBDA, N_CLASSES, LEARNING_RATE
)


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class DomainDiscriminator(nn.Module):
    """领域判别器"""

    def __init__(self, feature_dim: int, num_domains: int, hidden_dim: int = None):
        super(DomainDiscriminator, self).__init__()
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        self.hidden_dim = hidden_dim or DISCRIMINATOR_HIDDEN_DIM

        self.layers = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, num_domains)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, feature_dim)
        Returns:
            domain_logits: (batch_size, num_domains)
        """
        return self.layers(x)


class MMDLoss(nn.Module):
    """最大均值差异损失"""

    def __init__(self, kernel_type: str = 'rbf', kernel_mul: float = 2.0, kernel_num: int = 5):
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor,
                       kernel_mul: float, kernel_num: int) -> torch.Tensor:
        """计算高斯核矩阵"""
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        # 计算L2距离矩阵
        total0 = total.unsqueeze(0).expand(total.size(0), -1, -1)
        total1 = total.unsqueeze(1).expand(-1, total.size(0), -1)
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # 计算多个高斯核并求平均
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: (batch_size, feature_dim)
            target: (batch_size, feature_dim)
        Returns:
            mmd_loss: scalar
        """
        batch_size = source.size(0)
        kernels = self.gaussian_kernel(source, target, self.kernel_mul, self.kernel_num)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]

        loss = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)
        return loss


class MultiSourceDomainAdapter(nn.Module):
    """多源领域自适应模型"""

    def __init__(self, feature_extractor: nn.Module, source_domains: List[str],
                 feature_dim: int, n_classes: int = N_CLASSES):
        super(MultiSourceDomainAdapter, self).__init__()
        self.source_domains = source_domains
        self.num_sources = len(source_domains)
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.hidden_dim = MS_MDA_HIDDEN_DIM

        # 共享特征提取器
        self.shared_feature_extractor = feature_extractor

        # 为每个源域创建独立的适配分支
        self.domain_adapters = nn.ModuleDict()
        for domain in source_domains:
            self.domain_adapters[domain] = nn.Sequential(
                nn.Linear(feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

        # 每个源域的分类器
        self.domain_classifiers = nn.ModuleDict()
        for domain in source_domains:
            self.domain_classifiers[domain] = nn.Linear(self.hidden_dim, n_classes)

        # 集成分类器
        self.ensemble_classifier = nn.Linear(self.hidden_dim * self.num_sources, n_classes)

        # MMD损失计算
        self.mmd_loss = MMDLoss()

        # 领域权重（可学习）
        self.domain_weights = nn.Parameter(torch.ones(self.num_sources) / self.num_sources)

    def forward(self, x: torch.Tensor, domain: str = None, return_features: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Args:
            x: 输入数据
            domain: 数据来源域（训练时使用）
            return_features: 是否返回中间特征
        Returns:
            predictions 或 (predictions, features, domain_features)
        """
        # 共享特征提取
        # 如果特征提取器需要dataset_name参数（如UniversalFeatureSpace），则传递domain
        if hasattr(self.shared_feature_extractor, 'forward') and 'dataset_name' in self.shared_feature_extractor.forward.__code__.co_varnames:
            # 对于UniversalFeatureSpace，需要返回特征而不是分类结果
            if hasattr(self.shared_feature_extractor, 'forward') and 'return_features' in self.shared_feature_extractor.forward.__code__.co_varnames:
                shared_features = self.shared_feature_extractor(x, dataset_name=domain, return_features=True)
            else:
                shared_features = self.shared_feature_extractor(x, dataset_name=domain)
        else:
            shared_features = self.shared_feature_extractor(x)

        if domain is not None and domain in self.source_domains:
            # 训练模式：使用特定域的适配器
            adapted_features = self.domain_adapters[domain](shared_features)
            predictions = self.domain_classifiers[domain](adapted_features)

            if return_features:
                return predictions, shared_features, adapted_features
            return predictions

        else:
            # 测试模式：集成所有域的预测
            domain_features = []
            domain_predictions = []

            for domain_name in self.source_domains:
                adapted_features = self.domain_adapters[domain_name](shared_features)
                domain_pred = self.domain_classifiers[domain_name](adapted_features)
                domain_features.append(adapted_features)
                domain_predictions.append(domain_pred)

            # 特征级集成
            concatenated_features = torch.cat(domain_features, dim=1)
            ensemble_prediction = self.ensemble_classifier(concatenated_features)

            # 预测级集成
            if MS_MDA_ENSEMBLE_METHOD == 'weighted_average':
                weights = F.softmax(self.domain_weights, dim=0)
                weighted_predictions = sum(w * pred for w, pred in zip(weights, domain_predictions))
                final_prediction = 0.5 * ensemble_prediction + 0.5 * weighted_predictions
            elif MS_MDA_ENSEMBLE_METHOD == 'average':
                avg_prediction = sum(domain_predictions) / len(domain_predictions)
                final_prediction = 0.5 * ensemble_prediction + 0.5 * avg_prediction
            else:
                final_prediction = ensemble_prediction

            if return_features:
                return final_prediction, shared_features, domain_features
            return final_prediction

    def compute_adaptation_loss(self, source_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算域适应损失"""
        total_loss = 0
        num_pairs = 0

        # 计算所有源域对之间的MMD损失
        domains = list(source_features.keys())
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain1, domain2 = domains[i], domains[j]
                features1 = source_features[domain1]
                features2 = source_features[domain2]

                # 确保特征维度匹配
                min_batch_size = min(features1.size(0), features2.size(0))
                features1 = features1[:min_batch_size]
                features2 = features2[:min_batch_size]

                mmd = self.mmd_loss(features1, features2)
                total_loss += mmd
                num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=next(self.parameters()).device)


class AdversarialDomainAdapter(nn.Module):
    """对抗性领域自适应模型"""

    def __init__(self, feature_extractor: nn.Module, feature_dim: int,
                 num_domains: int, n_classes: int = N_CLASSES):
        super(AdversarialDomainAdapter, self).__init__()
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        self.n_classes = n_classes

        # 特征提取器
        self.feature_extractor = feature_extractor

        # 任务分类器
        self.task_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, n_classes)
        )

        # 领域判别器
        self.domain_discriminator = DomainDiscriminator(feature_dim, num_domains)

        # 梯度反转强度
        self.lambda_grl = GRADIENT_REVERSAL_LAMBDA

    def forward(self, x: torch.Tensor, alpha: float = 1.0, return_features: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Args:
            x: 输入数据
            alpha: 梯度反转强度调节参数
            return_features: 是否返回特征
        Returns:
            task_pred 或 (task_pred, domain_pred, features)
        """
        # 特征提取
        features = self.feature_extractor(x)

        # 任务预测
        task_pred = self.task_classifier(features)

        # 领域预测（带梯度反转）
        reversed_features = GradientReversalLayer.apply(features, alpha * self.lambda_grl)
        domain_pred = self.domain_discriminator(reversed_features)

        if return_features:
            return task_pred, domain_pred, features
        return task_pred

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取领域不变特征"""
        return self.feature_extractor(x)


class DomainAdaptationLoss(nn.Module):
    """领域自适应损失函数"""

    def __init__(self, adaptation_method: str):
        super(DomainAdaptationLoss, self).__init__()
        self.adaptation_method = adaptation_method
        self.task_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.CrossEntropyLoss()

    def forward(self, predictions: Dict, targets: Dict, model: nn.Module = None) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失

        Args:
            predictions: 预测结果字典
            targets: 目标字典
            model: 模型实例

        Returns:
            total_loss, loss_components
        """
        loss_components = {}

        # 任务损失
        task_loss = self.task_loss(predictions['task'], targets['task'])
        loss_components['task_loss'] = task_loss
        total_loss = task_loss

        if self.adaptation_method == 'ms_mda':
            # MS-MDA适应损失
            if 'adaptation' in predictions and model is not None:
                adaptation_loss = predictions['adaptation']
                loss_components['adaptation_loss'] = adaptation_loss
                total_loss += MS_MDA_ADAPTATION_WEIGHT * adaptation_loss

        elif self.adaptation_method == 'adversarial':
            # 对抗性损失
            if 'domain' in predictions and 'domain' in targets:
                domain_loss = self.domain_loss(predictions['domain'], targets['domain'])
                loss_components['domain_loss'] = domain_loss
                total_loss += ADVERSARIAL_WEIGHT * domain_loss

        loss_components['total_loss'] = total_loss
        return total_loss, loss_components


class DomainAdapterFactory:
    """领域适应器工厂类"""

    @staticmethod
    def create_domain_adapter(adaptation_method: str, feature_extractor: nn.Module,
                            feature_dim: int, domains_info: Dict, **kwargs) -> nn.Module:
        """
        创建领域适应器

        Args:
            adaptation_method: 适应方法名称
            feature_extractor: 特征提取器
            feature_dim: 特征维度
            domains_info: 域信息

        Returns:
            领域适应器实例
        """
        if adaptation_method == 'ms_mda':
            source_domains = list(domains_info.keys())
            return MultiSourceDomainAdapter(
                feature_extractor=feature_extractor,
                source_domains=source_domains,
                feature_dim=feature_dim,
                **kwargs
            )

        elif adaptation_method == 'adversarial':
            num_domains = len(domains_info)
            return AdversarialDomainAdapter(
                feature_extractor=feature_extractor,
                feature_dim=feature_dim,
                num_domains=num_domains,
                **kwargs
            )

        elif adaptation_method == 'none':
            # 返回原始特征提取器（无领域适应）
            return feature_extractor

        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")

    @staticmethod
    def create_optimizers(model: nn.Module, adaptation_method: str) -> Dict[str, torch.optim.Optimizer]:
        """
        为不同组件创建优化器

        Args:
            model: 模型实例
            adaptation_method: 适应方法

        Returns:
            优化器字典
        """
        optimizers = {}

        if adaptation_method == 'adversarial' and isinstance(model, AdversarialDomainAdapter):
            # 对抗性训练需要分别优化特征提取器和判别器
            feature_params = list(model.feature_extractor.parameters()) + list(model.task_classifier.parameters())
            discriminator_params = list(model.domain_discriminator.parameters())

            optimizers['feature'] = torch.optim.Adam(feature_params, lr=LEARNING_RATE)
            optimizers['discriminator'] = torch.optim.Adam(discriminator_params, lr=DISCRIMINATOR_LEARNING_RATE)

        else:
            # 其他情况使用统一优化器
            optimizers['main'] = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        return optimizers


class DomainAwareDataLoader:
    """领域感知数据加载器"""

    def __init__(self, datasets: Dict[str, torch.utils.data.Dataset], batch_size: int, shuffle: bool = True):
        self.datasets = datasets
        self.domain_names = list(datasets.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 为每个域创建数据加载器
        self.loaders = {}
        for domain, dataset in datasets.items():
            self.loaders[domain] = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle
            )

        # 创建迭代器
        self.iterators = {domain: iter(loader) for domain, loader in self.loaders.items()}

    def get_batch(self, domain: str = None) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        获取一个批次的数据

        Args:
            domain: 指定域名，None表示随机选择

        Returns:
            (data, labels, domain_name)
        """
        if domain is None:
            domain = np.random.choice(self.domain_names)

        try:
            data, labels = next(self.iterators[domain])
        except StopIteration:
            # 重新创建迭代器
            self.iterators[domain] = iter(self.loaders[domain])
            data, labels = next(self.iterators[domain])

        return data, labels, domain

    def get_mixed_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取混合域的批次数据

        Returns:
            (data, labels, domain_labels)
        """
        all_data = []
        all_labels = []
        all_domain_labels = []

        for domain_idx, domain in enumerate(self.domain_names):
            try:
                data, labels = next(self.iterators[domain])
            except StopIteration:
                self.iterators[domain] = iter(self.loaders[domain])
                data, labels = next(self.iterators[domain])

            all_data.append(data)
            all_labels.append(labels)
            all_domain_labels.append(torch.full((data.size(0),), domain_idx, dtype=torch.long))

        # 合并所有数据
        mixed_data = torch.cat(all_data, dim=0)
        mixed_labels = torch.cat(all_labels, dim=0)
        mixed_domain_labels = torch.cat(all_domain_labels, dim=0)

        return mixed_data, mixed_labels, mixed_domain_labels

    def reset(self):
        """重置所有迭代器"""
        self.iterators = {domain: iter(loader) for domain, loader in self.loaders.items()}