"""
Multi-dataset fusion methods for EEG signal processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math

from electrode_utils import ElectrodeGraphBuilder, create_positional_encoding, get_electrode_positions, create_unified_electrode_space
from config import (
    GCN_HIDDEN_DIM, GCN_NUM_LAYERS, GCN_EMBEDDING_DIM, GCN_DROPOUT,
    INPUT_WINDOW_SAMPLES, N_CLASSES
)


class GraphConvLayer(nn.Module):
    """图卷积层"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (batch_size, n_nodes, in_features)
            adj: (n_nodes, n_nodes)
        Returns:
            output: (batch_size, n_nodes, out_features)
        """
        support = torch.matmul(input, self.weight)  # (batch_size, n_nodes, out_features)
        output = torch.matmul(adj, support)  # (batch_size, n_nodes, out_features)
        if self.bias is not None:
            output = output + self.bias
        return output


class TemporalConvLayer(nn.Module):
    """时间卷积层"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_channels, n_timepoints)
        Returns:
            output: (batch_size, out_channels, n_timepoints)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SpatialTemporalGraphConv(nn.Module):
    """时空图卷积层"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(SpatialTemporalGraphConv, self).__init__()
        self.spatial_conv = GraphConvLayer(in_channels, out_channels)
        self.temporal_conv = TemporalConvLayer(out_channels, out_channels, kernel_size)
        self.dropout = nn.Dropout(GCN_DROPOUT)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_nodes, in_channels, n_timepoints)
            adj: (n_nodes, n_nodes)
        Returns:
            output: (batch_size, n_nodes, out_channels, n_timepoints)
        """
        batch_size, n_nodes, in_channels, n_timepoints = x.shape

        # 重塑为 (batch_size * n_timepoints, n_nodes, in_channels)
        x_reshaped = x.permute(0, 3, 1, 2).contiguous().view(-1, n_nodes, in_channels)

        # 空间图卷积
        x_spatial = self.spatial_conv(x_reshaped, adj)  # (batch_size * n_timepoints, n_nodes, out_channels)

        # 重塑回来并准备时间卷积
        x_spatial = x_spatial.view(batch_size, n_timepoints, n_nodes, -1)
        x_spatial = x_spatial.permute(0, 2, 3, 1).contiguous()  # (batch_size, n_nodes, out_channels, n_timepoints)

        # 对每个节点应用时间卷积
        outputs = []
        for i in range(n_nodes):
            node_data = x_spatial[:, i, :, :]  # (batch_size, out_channels, n_timepoints)
            temporal_out = self.temporal_conv(node_data)
            outputs.append(temporal_out.unsqueeze(1))  # (batch_size, 1, out_channels, n_timepoints)

        output = torch.cat(outputs, dim=1)  # (batch_size, n_nodes, out_channels, n_timepoints)
        output = self.dropout(output)

        return output


class GraphEEGEncoder(nn.Module):
    """图神经网络EEG编码器"""

    def __init__(self, n_channels: int, n_timepoints: int, embedding_dim: int = None):
        super(GraphEEGEncoder, self).__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.embedding_dim = embedding_dim or GCN_EMBEDDING_DIM

        # 图卷积层
        self.stgcn_layers = nn.ModuleList()
        current_channels = 1  # 每个电极作为1个通道输入

        for i in range(GCN_NUM_LAYERS):
            out_channels = GCN_HIDDEN_DIM if i < GCN_NUM_LAYERS - 1 else GCN_HIDDEN_DIM
            self.stgcn_layers.append(
                SpatialTemporalGraphConv(current_channels, out_channels)
            )
            current_channels = out_channels

        # 更有效的池化策略 - 保留更多的时空信息
        self.temporal_pool = nn.AdaptiveAvgPool1d(8)  # 保留8个时间点
        self.spatial_pool = nn.AdaptiveAvgPool1d(min(n_channels, 16))  # 保留关键空间信息

        # 预先计算特征维度并创建特征提取器
        # 最终特征维度 = n_channels * current_channels * 8
        final_feature_dim = n_channels * current_channels * 8

        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_feature_dim, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_channels, n_timepoints)
            adj: (n_channels, n_channels)
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        batch_size, n_channels, n_timepoints = x.shape

        # 重塑输入 (batch_size, n_channels, 1, n_timepoints)
        x = x.unsqueeze(2)

        # 通过图卷积层
        for layer in self.stgcn_layers:
            x = layer(x, adj)
            x = F.relu(x)

        # x shape: (batch_size, n_channels, out_channels, n_timepoints)
        batch_size, n_channels, out_channels, n_timepoints = x.shape

        # 更有效的池化：保留关键的时空信息
        # 对时间维度进行池化
        x = x.view(batch_size * n_channels, out_channels, n_timepoints)  # 重塑为3D进行时间池化
        x = self.temporal_pool(x)  # (batch_size * n_channels, out_channels, 8)

        # 重塑回4D并进行特征提取
        x = x.view(batch_size, n_channels, out_channels, 8)  # (batch_size, n_channels, out_channels, 8)

        # 特征提取
        embedding = self.feature_extractor(x)

        return embedding


class UniversalFeatureSpace(nn.Module):
    """通用特征空间融合模型"""

    def __init__(self, datasets_info: Dict[str, Dict], n_classes: int = N_CLASSES):
        super(UniversalFeatureSpace, self).__init__()
        self.datasets_info = datasets_info
        self.n_classes = n_classes
        self.embedding_dim = GCN_EMBEDDING_DIM

        # 为每个数据集创建图编码器
        self.encoders = nn.ModuleDict()
        self.graph_builder = ElectrodeGraphBuilder()

        for dataset_name, info in datasets_info.items():
            n_channels = len(info['channels'])
            n_timepoints = info.get('n_timepoints', INPUT_WINDOW_SAMPLES)
            self.encoders[dataset_name] = GraphEEGEncoder(n_channels, n_timepoints, self.embedding_dim)

        # 特征对齐层
        self.feature_alignment = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # 增强的分类头 - 增加复杂度确保充分训练
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim // 4),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(self.embedding_dim // 4, n_classes)
        )

        # 预构建图结构
        self.adjacency_matrices = {}
        for dataset_name, info in datasets_info.items():
            adj = self.graph_builder.build_graph(info['channels'], dataset_name)
            self.register_buffer(f'adj_{dataset_name}', adj)
            self.adjacency_matrices[dataset_name] = adj

    def forward(self, x: torch.Tensor, dataset_name: str, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_channels, n_timepoints)
            dataset_name: 数据集名称
            return_features: 是否返回特征而不是分类结果
        Returns:
            output: (batch_size, n_classes) 或 (batch_size, embedding_dim) 如果 return_features=True
        """
        # 获取对应的编码器和邻接矩阵
        encoder = self.encoders[dataset_name]
        adj = getattr(self, f'adj_{dataset_name}')

        # 确保邻接矩阵在正确的设备上
        adj = adj.to(x.device)

        # 编码为图特征
        embedding = encoder(x, adj)

        # 特征对齐
        aligned_features = self.feature_alignment(embedding)

        # 根据需要返回特征或分类结果
        if return_features:
            return aligned_features

        # 分类
        output = self.classifier(aligned_features)

        return output

    def extract_features(self, x: torch.Tensor, dataset_name: str) -> torch.Tensor:
        """提取对齐后的特征"""
        encoder = self.encoders[dataset_name]
        adj = getattr(self, f'adj_{dataset_name}')

        embedding = encoder(x, adj)
        aligned_features = self.feature_alignment(embedding)

        return aligned_features

    def forward_single_dataset(self, x: torch.Tensor, dataset_name: str = None) -> torch.Tensor:
        """
        用于单数据集测试的简化前向传播方法
        如果没有指定dataset_name，使用第一个可用的数据集
        """
        if dataset_name is None:
            dataset_name = list(self.datasets_info.keys())[0]
        return self.forward(x, dataset_name)


class LightweightGraphEnhancer(nn.Module):
    """轻量级图特征增强器 - 作为主干模型的预处理层"""

    def __init__(self, n_channels: int, enhancement_strength: float = 0.1):
        super(LightweightGraphEnhancer, self).__init__()
        self.n_channels = n_channels
        self.enhancement_strength = enhancement_strength

        # 轻量级图卷积层 - 只做一层简单的空间特征增强
        self.spatial_conv = GraphConvLayer(1, 1, bias=False)  # 输入输出都是1维特征

        # 可学习的增强强度门控
        self.enhancement_gate = nn.Parameter(torch.tensor(enhancement_strength))

        # 邻接矩阵缓存
        self.adjacency_matrix = None
        self.graph_builder = ElectrodeGraphBuilder()

    def build_graph_once(self, channels: List[str], dataset_name: str = 'combined'):
        """构建并缓存邻接矩阵（只在第一次调用时执行）"""
        if self.adjacency_matrix is None:
            adj = self.graph_builder.build_graph(channels, dataset_name)
            self.adjacency_matrix = adj
        return self.adjacency_matrix

    def forward(self, x: torch.Tensor, channels: List[str] = None) -> torch.Tensor:
        """
        轻量级图增强前向传播
        Args:
            x: (batch_size, n_channels, n_timepoints) - 原始EEG信号
            channels: 电极名称列表（可选，用于构建图）
        Returns:
            enhanced_x: (batch_size, n_channels, n_timepoints) - 增强后的EEG信号
        """
        batch_size, n_channels, n_timepoints = x.shape

        # 总是使用简单的局部连接图，维度与实际输入匹配
        adj = self._create_simple_adjacency(n_channels).to(x.device)

        # 计算增强强度（sigmoid确保在0-1之间）
        gate = torch.sigmoid(self.enhancement_gate)

        # 重塑输入进行批量图卷积：(batch_size, n_channels, n_timepoints) -> (batch_size * n_timepoints, n_channels, 1)
        x_reshaped = x.permute(0, 2, 1).contiguous()  # (batch_size, n_timepoints, n_channels)
        x_reshaped = x_reshaped.view(-1, n_channels, 1)  # (batch_size * n_timepoints, n_channels, 1)

        # 批量图卷积增强
        enhanced_reshaped = self.spatial_conv(x_reshaped, adj)  # (batch_size * n_timepoints, n_channels, 1)

        # 重塑回原始形状
        enhanced_reshaped = enhanced_reshaped.squeeze(-1)  # (batch_size * n_timepoints, n_channels)
        enhanced_x = enhanced_reshaped.view(batch_size, n_timepoints, n_channels)  # (batch_size, n_timepoints, n_channels)
        enhanced_x = enhanced_x.permute(0, 2, 1).contiguous()  # (batch_size, n_channels, n_timepoints)

        # 残差连接：原始信号 + 门控的增强特征
        output = x + gate * enhanced_x

        return output

    def _create_simple_adjacency(self, n_channels: int = None) -> torch.Tensor:
        """创建简单的局部连接邻接矩阵（当没有电极位置信息时）"""
        if n_channels is None:
            n_channels = self.n_channels

        adj = torch.eye(n_channels, dtype=torch.float32)

        # 添加相邻通道的连接（简单的一维邻接）
        for i in range(n_channels - 1):
            adj[i, i + 1] = 0.5
            adj[i + 1, i] = 0.5

        # 对称归一化：D^(-1/2) A D^(-1/2)
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

        # 创建度矩阵的逆平方根
        degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj = torch.mm(torch.mm(degree_matrix_inv_sqrt, adj), degree_matrix_inv_sqrt)

        return adj


class GraphEnhancedModel(nn.Module):
    """图增强 + 主干模型的混合架构"""

    def __init__(self, base_model_class, base_model_params: Dict,
                 datasets_info: Dict, enhancement_strength: float = 0.1):
        super(GraphEnhancedModel, self).__init__()

        # 创建统一的电极空间
        self.unified_channels, self.channel_mapping = create_unified_electrode_space(datasets_info)
        self.unified_n_channels = len(self.unified_channels)

        print(f"GraphEnhancedModel: 使用统一电极空间，{self.unified_n_channels}个电极")

        # 图特征增强器 - 使用统一的电极空间
        self.graph_enhancer = LightweightGraphEnhancer(
            n_channels=self.unified_n_channels,
            enhancement_strength=enhancement_strength
        )

        # 主干模型（如SepConv1D）- 使用统一的通道数
        unified_model_params = base_model_params.copy()
        unified_model_params['n_chans'] = self.unified_n_channels
        self.backbone = base_model_class(**unified_model_params)

    def forward(self, x: torch.Tensor, channels: List[str] = None, dataset_name: str = None, **kwargs) -> torch.Tensor:
        """
        混合模型前向传播
        Args:
            x: (batch_size, n_channels, n_timepoints)
            channels: 电极名称列表（可选）
            dataset_name: 数据集名称，用于电极映射
            **kwargs: 其他参数（为了兼容性）
        Returns:
            output: (batch_size, n_classes)
        """
        batch_size, input_channels, n_timepoints = x.shape

        # 第一步：将输入映射到统一的电极空间
        if dataset_name and dataset_name in self.channel_mapping:
            # 使用数据集特定的映射
            channel_map = self.channel_mapping[dataset_name]
            mapped_x = self._map_to_unified_space(x, channel_map, batch_size, n_timepoints)
        else:
            # 回退到简单的填充/裁剪策略
            mapped_x = self._simple_resize(x, batch_size, n_timepoints)

        # 第二步：图特征增强
        enhanced_x = self.graph_enhancer(mapped_x, self.unified_channels)

        # 第三步：主干模型分类
        output = self.backbone(enhanced_x)

        return output

    def _map_to_unified_space(self, x: torch.Tensor, channel_map: Dict[int, int],
                            batch_size: int, n_timepoints: int) -> torch.Tensor:
        """将输入映射到统一的电极空间"""
        # 创建零填充的统一空间张量
        unified_x = torch.zeros(batch_size, self.unified_n_channels, n_timepoints,
                              dtype=x.dtype, device=x.device)

        # 根据映射填充数据
        for input_idx, unified_idx in channel_map.items():
            if input_idx < x.shape[1]:  # 确保输入索引有效
                unified_x[:, unified_idx, :] = x[:, input_idx, :]

        return unified_x

    def _simple_resize(self, x: torch.Tensor, batch_size: int, n_timepoints: int) -> torch.Tensor:
        """简单的尺寸调整策略（回退方案）"""
        input_channels = x.shape[1]

        if input_channels == self.unified_n_channels:
            return x
        elif input_channels < self.unified_n_channels:
            # 零填充
            padding = torch.zeros(batch_size, self.unified_n_channels - input_channels, n_timepoints,
                                dtype=x.dtype, device=x.device)
            return torch.cat([x, padding], dim=1)
        else:
            # 裁剪
            return x[:, :self.unified_n_channels, :]

    def extract_features(self, x: torch.Tensor, channels: List[str] = None) -> torch.Tensor:
        """提取增强后的特征"""
        enhanced_x = self.graph_enhancer(x, channels)

        # 如果主干模型有特征提取方法
        if hasattr(self.backbone, 'extract_features'):
            features = self.backbone.extract_features(enhanced_x)
        else:
            # 否则返回增强后的原始信号特征
            features = enhanced_x.view(enhanced_x.shape[0], -1)

        return features


class FusionModelFactory:
    """融合模型工厂类"""

    @staticmethod
    def create_fusion_model(fusion_method: str, datasets_info: Dict, base_model_info: Dict = None) -> nn.Module:
        """
        创建融合模型

        Args:
            fusion_method: 融合方法名称
            datasets_info: 数据集信息
            base_model_info: 基础模型信息

        Returns:
            融合模型实例
        """
        if fusion_method == 'graph_gcn':
            return UniversalFeatureSpace(datasets_info)

        elif fusion_method == 'graph_enhanced':
            # 新的混合方法：轻量级GCN增强 + 主干模型
            if base_model_info is None:
                raise ValueError("base_model_info is required for graph_enhanced method")

            return GraphEnhancedModel(
                base_model_class=base_model_info['class'],
                base_model_params=base_model_info['params'],
                datasets_info=datasets_info,
                enhancement_strength=0.1  # 可配置的增强强度
            )

        elif fusion_method == 'none':
            # 返回None，表示使用原始baseline方法
            return None

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    @staticmethod
    def get_position_tensors(datasets_info: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        """
        获取各数据集的电极位置张量

        Args:
            datasets_info: 数据集信息

        Returns:
            {dataset_name: position_tensor}
        """
        position_tensors = {}

        for dataset_name, info in datasets_info.items():
            channels = info['channels']
            positions = get_electrode_positions(channels, dataset_name)

            # 转换为张量
            pos_array = np.array([positions[ch] for ch in channels])
            position_tensors[dataset_name] = torch.FloatTensor(pos_array)

        return position_tensors