"""
Multi-dataset fusion methods for EEG signal processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math

from electrode_utils import ElectrodeGraphBuilder, create_positional_encoding, get_electrode_positions
from config import (
    GCN_HIDDEN_DIM, GCN_NUM_LAYERS, GCN_EMBEDDING_DIM, GCN_DROPOUT,
    SPATIAL_ATTENTION_VIRTUAL_CHANNELS, SPATIAL_ATTENTION_HIDDEN_DIM,
    SPATIAL_ATTENTION_NUM_HEADS, SPATIAL_ATTENTION_DROPOUT,
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

        # 全局池化和特征提取
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 我们需要推断最终的特征维度，先设置为None，在第一次前向传播时确定
        self.feature_extractor = None
        self.embedding_dim = embedding_dim or GCN_EMBEDDING_DIM

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_channels, n_timepoints)
            adj: (n_channels, n_channels)
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        batch_size = x.shape[0]

        # 重塑输入 (batch_size, n_channels, 1, n_timepoints)
        x = x.unsqueeze(2)

        # 通过图卷积层
        for layer in self.stgcn_layers:
            x = layer(x, adj)
            x = F.relu(x)

        # 全局池化
        x = self.global_pool(x)  # (batch_size, n_channels, 1, 1)
        x = x.view(batch_size, -1)  # (batch_size, flattened_features)

        # 动态创建特征提取器（仅在第一次运行时）
        if self.feature_extractor is None:
            input_dim = x.shape[1]
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(GCN_DROPOUT),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            ).to(x.device)

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

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(self.embedding_dim // 2, n_classes)
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


class SpatialAttentionLayer(nn.Module):
    """空间注意力层"""

    def __init__(self, max_channels: int, virtual_channels: int, hidden_dim: int = None):
        super(SpatialAttentionLayer, self).__init__()
        self.max_channels = max_channels
        self.virtual_channels = virtual_channels
        self.hidden_dim = hidden_dim or SPATIAL_ATTENTION_HIDDEN_DIM

        # 位置编码维度
        self.pos_encoding_dim = 64

        # 空间位置编码网络
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.pos_encoding_dim),  # 3D坐标输入
            nn.ReLU(),
            nn.Linear(self.pos_encoding_dim, self.pos_encoding_dim)
        )

        # 多头注意力机制
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.pos_encoding_dim,
            num_heads=SPATIAL_ATTENTION_NUM_HEADS,
            dropout=SPATIAL_ATTENTION_DROPOUT,
            batch_first=True
        )

        # 权重生成网络
        self.weight_generator = nn.Sequential(
            nn.Linear(self.pos_encoding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(SPATIAL_ATTENTION_DROPOUT),
            nn.Linear(self.hidden_dim, virtual_channels)
        )

        # 虚拟通道的可学习位置
        self.virtual_positions = nn.Parameter(
            torch.randn(virtual_channels, 3) * 0.1
        )

    def forward(self, x: torch.Tensor, electrode_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_channels, n_timepoints)
            electrode_positions: (n_channels, 3) - 3D电极位置
        Returns:
            output: (batch_size, virtual_channels, n_timepoints)
        """
        batch_size, n_channels, n_timepoints = x.shape

        # 编码电极位置
        real_pos_encoded = self.position_encoder(electrode_positions)  # (n_channels, pos_encoding_dim)
        virtual_pos_encoded = self.position_encoder(self.virtual_positions)  # (virtual_channels, pos_encoding_dim)

        # 扩展批次维度
        real_pos_encoded = real_pos_encoded.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, n_channels, pos_encoding_dim)
        virtual_pos_encoded = virtual_pos_encoded.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, virtual_channels, pos_encoding_dim)

        # 计算注意力权重 (虚拟 -> 真实位置)
        attention_output, attention_weights = self.multihead_attention(
            query=virtual_pos_encoded,  # (batch_size, V, D)
            key=real_pos_encoded,      # (batch_size, C, D)
            value=real_pos_encoded     # (batch_size, C, D)
        )
        # 直接使用注意力权重进行通道融合（注意力权重已沿 key 维度 softmax）
        # attention_weights: (B, V, C)
        weights_matrix = attention_weights

        # 将实通道信号 x (B, C, T) 线性组合为虚拟通道 (B, V, T)
        # 使用批处理矩阵乘法: (B, V, C) @ (B, C, T) -> (B, V, T)
        output = torch.matmul(weights_matrix, x)

        return output


class SpatialAttentionModel(nn.Module):
    """基于空间注意力的端到端协调模型"""

    def __init__(self, max_channels: int, base_model_class, base_model_params: Dict, n_classes: int = N_CLASSES):
        super(SpatialAttentionModel, self).__init__()
        self.max_channels = max_channels
        self.virtual_channels = SPATIAL_ATTENTION_VIRTUAL_CHANNELS
        self.n_classes = n_classes

        # 空间注意力前端
        self.spatial_attention = SpatialAttentionLayer(max_channels, self.virtual_channels)

        # 修改基础模型参数以适应虚拟通道
        modified_params = base_model_params.copy()
        if 'n_chans' in modified_params:
            modified_params['n_chans'] = self.virtual_channels
        if 'n_channels' in modified_params:
            modified_params['n_channels'] = self.virtual_channels

        # 基础分类器后端
        self.base_model = base_model_class(**modified_params)

    def forward(self, x: torch.Tensor, electrode_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_channels, n_timepoints)
            electrode_positions: (n_channels, 3)
        Returns:
            output: (batch_size, n_classes)
        """
        # 空间注意力变换
        x_transformed = self.spatial_attention(x, electrode_positions)

        # 基础模型分类
        output = self.base_model(x_transformed)

        return output

    def extract_features(self, x: torch.Tensor, electrode_positions: torch.Tensor) -> torch.Tensor:
        """提取空间变换后的特征"""
        x_transformed = self.spatial_attention(x, electrode_positions)

        # 如果基础模型有特征提取方法
        if hasattr(self.base_model, 'extract_features'):
            features = self.base_model.extract_features(x_transformed)
        else:
            # 否则使用变换后的原始信号作为特征
            features = x_transformed.view(x_transformed.shape[0], -1)

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

        elif fusion_method == 'spatial_attention':
            if base_model_info is None:
                raise ValueError("base_model_info is required for spatial_attention method")

            max_channels = max(len(info['channels']) for info in datasets_info.values())
            return SpatialAttentionModel(
                max_channels=max_channels,
                base_model_class=base_model_info['class'],
                base_model_params=base_model_info['params']
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