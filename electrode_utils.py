"""
Electrode position management and graph construction utilities
"""

import numpy as np
import torch
import scipy.spatial.distance as dist
from typing import Dict, List, Tuple, Optional
import mne

# 标准10-20系统电极位置 (以Cz为原点的相对坐标，单位: cm)
STANDARD_ELECTRODE_POSITIONS = {
    # 前额区
    'Fp1': (-2.7, 8.8, 2.4), 'Fp2': (2.7, 8.8, 2.4), 'Fpz': (0.0, 9.0, 2.6),
    'F7': (-6.1, 4.9, 6.6), 'F3': (-4.0, 5.4, 7.8), 'Fz': (0.0, 6.0, 8.5),
    'F4': (4.0, 5.4, 7.8), 'F8': (6.1, 4.9, 6.6),
    'FT7': (-7.5, 2.4, 4.0), 'FC5': (-5.6, 3.2, 6.8), 'FC1': (-2.0, 4.2, 8.3),
    'FCz': (0.0, 4.5, 8.5), 'FC2': (2.0, 4.2, 8.3), 'FC6': (5.6, 3.2, 6.8),
    'FT8': (7.5, 2.4, 4.0),

    # 中央区
    'T7': (-8.0, 0.0, 1.5), 'C5': (-6.0, 0.0, 6.5), 'C3': (-4.2, 0.0, 8.0),
    'C1': (-2.1, 0.0, 8.7), 'Cz': (0.0, 0.0, 9.0), 'C2': (2.1, 0.0, 8.7),
    'C4': (4.2, 0.0, 8.0), 'C6': (6.0, 0.0, 6.5), 'T8': (8.0, 0.0, 1.5),

    # 顶区
    'TP7': (-7.5, -2.4, 4.0), 'CP5': (-5.6, -3.2, 6.8), 'CP1': (-2.0, -4.2, 8.3),
    'CPz': (0.0, -4.5, 8.5), 'CP2': (2.0, -4.2, 8.3), 'CP6': (5.6, -3.2, 6.8),
    'TP8': (7.5, -2.4, 4.0),
    'P7': (-6.1, -4.9, 6.6), 'P3': (-4.0, -5.4, 7.8), 'Pz': (0.0, -6.0, 8.5),
    'P4': (4.0, -5.4, 7.8), 'P8': (6.1, -4.9, 6.6),
    'PO7': (-5.0, -7.5, 5.5), 'PO3': (-2.5, -7.8, 7.0), 'POz': (0.0, -8.0, 7.5),
    'PO4': (2.5, -7.8, 7.0), 'PO8': (5.0, -7.5, 5.5),

    # 枕区
    'O1': (-2.7, -8.8, 2.4), 'Oz': (0.0, -9.0, 2.6), 'O2': (2.7, -8.8, 2.4),

    # 额外电极
    'AF7': (-4.5, 7.5, 4.8), 'AF3': (-2.5, 7.8, 6.2), 'AFz': (0.0, 8.0, 6.5),
    'AF4': (2.5, 7.8, 6.2), 'AF8': (4.5, 7.5, 4.8),
    'F1': (-2.0, 5.8, 8.2), 'F2': (2.0, 5.8, 8.2),
    'F5': (-5.0, 4.8, 7.2), 'F6': (5.0, 4.8, 7.2),
    'FT9': (-8.5, 3.0, 1.8), 'FT10': (8.5, 3.0, 1.8),
    'T9': (-8.5, 0.0, 0.5), 'T10': (8.5, 0.0, 0.5),
    'TP9': (-8.5, -3.0, 1.8), 'TP10': (8.5, -3.0, 1.8),
    'P1': (-2.0, -5.8, 8.2), 'P2': (2.0, -5.8, 8.2),
    'P5': (-5.0, -4.8, 7.2), 'P6': (5.0, -4.8, 7.2), 'P9': (-7.0, -6.5, 3.5), 'P10': (7.0, -6.5, 3.5),
}

# 数据集特定的电极映射
DATASET_ELECTRODE_MAPPINGS = {
    'P3': {
        # P3数据集的电极名称映射到标准位置
        'FP1': 'Fp1', 'FP2': 'Fp2',
        # 其他映射...
    },
    'AVO': {
        # AVO数据集的电极名称映射
        'Fp1': 'Fp1', 'Fp2': 'Fp2',
        # 其他映射...
    }
}


def get_electrode_positions(channels: List[str], dataset: str = 'standard') -> Dict[str, Tuple[float, float, float]]:
    """
    获取指定通道的3D电极位置

    Args:
        channels: 电极通道名称列表
        dataset: 数据集名称，用于电极名称映射

    Returns:
        电极位置字典 {channel_name: (x, y, z)}
    """
    positions = {}
    mapping = DATASET_ELECTRODE_MAPPINGS.get(dataset, {})

    for channel in channels:
        # 应用数据集特定映射
        standard_name = mapping.get(channel, channel)

        # 尝试多种名称变体
        for name_variant in [standard_name, standard_name.upper(), standard_name.lower(),
                           standard_name.capitalize()]:
            if name_variant in STANDARD_ELECTRODE_POSITIONS:
                positions[channel] = STANDARD_ELECTRODE_POSITIONS[name_variant]
                break
        else:
            # 如果找不到，使用MNE的标准位置
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                if channel in montage.ch_names:
                    pos = montage.get_positions()['ch_pos'][channel]
                    positions[channel] = (pos[0] * 100, pos[1] * 100, pos[2] * 100)  # 转换为cm
                else:
                    # 对于特征通道（feat_*），静默处理，不显示警告
                    if not channel.startswith('feat_'):
                        print(f"Warning: Position for electrode {channel} not found, using default")
                    positions[channel] = (0.0, 0.0, 0.0)
            except:
                # 对于特征通道（feat_*），静默处理，不显示警告
                if not channel.startswith('feat_'):
                    print(f"Warning: Position for electrode {channel} not found, using default")
                positions[channel] = (0.0, 0.0, 0.0)

    return positions


def calculate_distance_matrix(positions: Dict[str, Tuple[float, float, float]]) -> np.ndarray:
    """
    计算电极间的欧式距离矩阵

    Args:
        positions: 电极位置字典

    Returns:
        距离矩阵 (n_electrodes, n_electrodes)
    """
    channels = list(positions.keys())
    coords = np.array([positions[ch] for ch in channels])

    # 计算欧式距离矩阵
    distance_matrix = dist.cdist(coords, coords, metric='euclidean')

    return distance_matrix


def create_adjacency_matrix(distance_matrix: np.ndarray,
                          method: str = 'knn',
                          k: int = 4,
                          threshold: float = 3.0) -> np.ndarray:
    """
    根据距离矩阵创建邻接矩阵

    Args:
        distance_matrix: 电极间距离矩阵
        method: 邻接矩阵构建方法 ('knn', 'threshold', 'gaussian')
        k: KNN方法的邻居数
        threshold: 阈值方法的距离阈值(cm)

    Returns:
        邻接矩阵 (n_electrodes, n_electrodes)
    """
    n_electrodes = distance_matrix.shape[0]
    adjacency = np.zeros_like(distance_matrix)

    if method == 'knn':
        # K最近邻方法
        for i in range(n_electrodes):
            # 找到k个最近邻居 (排除自己)
            distances = distance_matrix[i]
            nearest_indices = np.argsort(distances)[1:k+1]
            adjacency[i, nearest_indices] = 1
            adjacency[nearest_indices, i] = 1  # 对称化

    elif method == 'threshold':
        # 距离阈值方法
        adjacency = (distance_matrix <= threshold).astype(float)
        np.fill_diagonal(adjacency, 0)  # 移除自连接

    elif method == 'gaussian':
        # 高斯权重方法
        sigma = threshold  # 使用threshold作为高斯核的标准差
        adjacency = np.exp(-distance_matrix**2 / (2 * sigma**2))
        np.fill_diagonal(adjacency, 0)  # 移除自连接

    return adjacency


def create_electrode_graph(channels: List[str],
                         dataset: str = 'standard',
                         adjacency_method: str = 'knn',
                         **kwargs) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    为给定电极创建图结构

    Args:
        channels: 电极通道列表
        dataset: 数据集名称
        adjacency_method: 邻接矩阵构建方法
        **kwargs: 邻接矩阵方法的额外参数

    Returns:
        adjacency_matrix: 邻接矩阵
        channel_to_idx: 通道名到索引的映射
    """
    # 获取电极位置
    positions = get_electrode_positions(channels, dataset)

    # 计算距离矩阵
    distance_matrix = calculate_distance_matrix(positions)

    # 创建邻接矩阵
    adjacency_matrix = create_adjacency_matrix(distance_matrix, adjacency_method, **kwargs)

    # 创建通道到索引的映射
    channel_to_idx = {ch: i for i, ch in enumerate(channels)}

    return adjacency_matrix, channel_to_idx


def normalize_adjacency_matrix(adjacency: np.ndarray, method: str = 'symmetric') -> np.ndarray:
    """
    归一化邻接矩阵

    Args:
        adjacency: 原始邻接矩阵
        method: 归一化方法 ('symmetric', 'random_walk', 'none')

    Returns:
        归一化后的邻接矩阵
    """
    if method == 'none':
        return adjacency

    # 添加自连接
    adjacency_with_self = adjacency + np.eye(adjacency.shape[0])

    if method == 'symmetric':
        # 对称归一化: D^(-1/2) * A * D^(-1/2)
        degree = np.sum(adjacency_with_self, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        degree_matrix_inv_sqrt = np.diag(degree_inv_sqrt)
        normalized = degree_matrix_inv_sqrt @ adjacency_with_self @ degree_matrix_inv_sqrt

    elif method == 'random_walk':
        # 随机游走归一化: D^(-1) * A
        degree = np.sum(adjacency_with_self, axis=1)
        degree_inv = np.power(degree, -1)
        degree_inv[np.isinf(degree_inv)] = 0
        degree_matrix_inv = np.diag(degree_inv)
        normalized = degree_matrix_inv @ adjacency_with_self

    return normalized


class ElectrodeGraphBuilder:
    """电极图构建器类"""

    def __init__(self, adjacency_method: str = 'knn', normalize_method: str = 'symmetric'):
        self.adjacency_method = adjacency_method
        self.normalize_method = normalize_method
        self.graphs_cache = {}  # 缓存已构建的图

    def build_graph(self, channels: List[str], dataset: str = 'standard', **kwargs) -> torch.Tensor:
        """
        构建电极图的邻接矩阵

        Args:
            channels: 电极通道列表
            dataset: 数据集名称
            **kwargs: 邻接矩阵构建的额外参数

        Returns:
            归一化的邻接矩阵 (torch.Tensor)
        """
        # 创建缓存键
        cache_key = (tuple(sorted(channels)), dataset, self.adjacency_method, str(kwargs))

        if cache_key in self.graphs_cache:
            return self.graphs_cache[cache_key]

        # 构建图
        adjacency, channel_to_idx = create_electrode_graph(
            channels, dataset, self.adjacency_method, **kwargs
        )

        # 归一化
        normalized_adjacency = normalize_adjacency_matrix(adjacency, self.normalize_method)

        # 转换为torch tensor
        adjacency_tensor = torch.FloatTensor(normalized_adjacency)

        # 缓存结果
        self.graphs_cache[cache_key] = adjacency_tensor

        return adjacency_tensor

    def get_multi_dataset_graphs(self, datasets_info: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        为多个数据集构建图

        Args:
            datasets_info: {dataset_name: [channels]}

        Returns:
            {dataset_name: adjacency_tensor}
        """
        graphs = {}
        for dataset_name, channels in datasets_info.items():
            graphs[dataset_name] = self.build_graph(channels, dataset_name)
        return graphs


# 用于空间注意力的位置编码
def create_positional_encoding(positions: Dict[str, Tuple[float, float, float]],
                             embedding_dim: int) -> torch.Tensor:
    """
    为电极位置创建位置编码

    Args:
        positions: 电极位置字典
        embedding_dim: 编码维度

    Returns:
        位置编码矩阵 (n_electrodes, embedding_dim)
    """
    channels = list(positions.keys())
    n_electrodes = len(channels)

    # 提取3D坐标
    coords = np.array([positions[ch] for ch in channels])  # (n_electrodes, 3)

    # 归一化坐标到[-1, 1]
    coords_normalized = 2 * (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0)) - 1

    # 创建位置编码
    encoding = np.zeros((n_electrodes, embedding_dim))

    # 使用正弦和余弦编码
    for i in range(3):  # x, y, z坐标
        for j in range(embedding_dim // 6):  # 每个坐标使用embedding_dim//6的维度
            pos = coords_normalized[:, i]
            encoding[:, i * (embedding_dim // 6) + j] = np.sin(pos / (10000 ** (2 * j / embedding_dim)))
            if i * (embedding_dim // 6) + j + embedding_dim // 6 < embedding_dim:
                encoding[:, i * (embedding_dim // 6) + j + embedding_dim // 6] = np.cos(pos / (10000 ** (2 * j / embedding_dim)))

    return torch.FloatTensor(encoding)


def interpolate_to_common_space(data: np.ndarray,
                              source_channels: List[str],
                              target_channels: List[str],
                              source_dataset: str = 'standard',
                              target_dataset: str = 'standard') -> np.ndarray:
    """
    将EEG数据从源电极布局插值到目标电极布局

    Args:
        data: 源数据 (n_samples, n_source_channels, n_timepoints)
        source_channels: 源电极通道
        target_channels: 目标电极通道
        source_dataset: 源数据集名称
        target_dataset: 目标数据集名称

    Returns:
        插值后的数据 (n_samples, n_target_channels, n_timepoints)
    """
    # 获取电极位置
    source_positions = get_electrode_positions(source_channels, source_dataset)
    target_positions = get_electrode_positions(target_channels, target_dataset)

    # 提取坐标
    source_coords = np.array([source_positions[ch] for ch in source_channels])
    target_coords = np.array([target_positions[ch] for ch in target_channels])

    # 计算插值权重矩阵
    from scipy.interpolate import griddata

    n_samples, n_source, n_timepoints = data.shape
    interpolated_data = np.zeros((n_samples, len(target_channels), n_timepoints))

    # 对每个时间点进行插值
    for t in range(n_timepoints):
        for s in range(n_samples):
            # 当前时间点的数据
            values = data[s, :, t]

            # 使用三次插值
            interpolated_values = griddata(
                source_coords, values, target_coords,
                method='cubic', fill_value=0.0
            )

            interpolated_data[s, :, t] = interpolated_values

    return interpolated_data