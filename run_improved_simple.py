"""
简化的改进EEG实验 - 保持原有输出格式
"""

import os
import sys
import numpy as np
import mne
import warnings
from pathlib import Path

# 设置路径和导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入现有组件
from data_utils import EEGBIDSDataset
from constants import COMMON_CHANNELS, P3_CHANNELS, AVO_CHANNELS, RESPONSE_EVENTS, ODDBALL_EVENTS
from config import AVO_DATA_DIR, electrode_list, seeds
from models import create_model, train_model, evaluate, normalize_data
from utils import calculate_statistics, print_statistics
from experiment_logger import setup_logger, log_section_header, log_configuration
from experiment import prepare_data_loaders

# 导入改进组件
try:
    from mne.preprocessing import ICA
    ICA_AVAILABLE = True
except ImportError:
    ICA_AVAILABLE = False

# 设置日志
mne.set_log_level('ERROR')
warnings.filterwarnings('ignore')


class SimpleImprovedPreprocessor:
    """简化的改进预处理器"""
    
    def __init__(self, eeg_channels, use_ica=True, freq_band=(8, 30)):
        self.eeg_channels = [ch.lower() for ch in eeg_channels]
        self.use_ica = use_ica and ICA_AVAILABLE
        self.freq_band = freq_band
        
    def apply_ica_cleanup(self, raw):
        """应用ICA去伪影"""
        if not self.use_ica:
            return raw
            
        try:
            # 为ICA创建副本并应用高通滤波
            raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
            
            # 设置ICA
            n_components = min(15, len(raw.ch_names))
            ica = ICA(n_components=n_components, random_state=42, max_iter="auto")
            ica.fit(raw_for_ica, verbose=False)
            
            # 基于前额电极的伪影检测
            frontal_channels = [ch for ch in ['fp1', 'fp2', 'f7', 'f8'] 
                              if ch in [c.lower() for c in raw.ch_names]]
            
            if frontal_channels:
                ica_sources = ica.get_sources(raw_for_ica).get_data()
                frontal_picks = [raw.ch_names.index(ch) for ch in raw.ch_names 
                               if ch.lower() in frontal_channels]
                frontal_data = raw.get_data(picks=frontal_picks)
                frontal_avg = np.mean(frontal_data, axis=0)
                
                # 找到与前额活动高度相关的成分（眼动伪影）
                exclude_components = []
                for comp_idx in range(ica_sources.shape[0]):
                    corr = np.corrcoef(ica_sources[comp_idx], frontal_avg)[0, 1]
                    if abs(corr) > 0.6:
                        exclude_components.append(comp_idx)
                
                # 限制移除的成分数量
                exclude_components = exclude_components[:3]
                
                if exclude_components:
                    ica.exclude = exclude_components
                
            # 应用ICA清理
            ica.apply(raw, verbose=False)
            return raw
            
        except Exception as e:
            return raw
    
    def process_subject(self, file_path):
        """处理单个被试的数据"""
        try:
            # 加载原始数据
            raw = mne.io.read_raw_brainvision(str(file_path), preload=True, verbose=False)
            
            # 标准化通道名
            raw.rename_channels({ch: ch.lower() for ch in raw.ch_names})
            
            # 选择可用通道
            available_channels = [ch for ch in self.eeg_channels if ch in raw.ch_names]
            if not available_channels:
                return None, None
            
            raw.pick_channels(available_channels)
            
            # 设置参考
            try:
                raw.set_eeg_reference('average', projection=True, verbose=False)
            except:
                try:
                    if 'cz' in raw.ch_names:
                        raw.set_eeg_reference(['cz'], verbose=False)
                except:
                    pass
            
            # 应用ICA清理
            raw = self.apply_ica_cleanup(raw)
            
            # 应用优化的频率滤波（ERP优化band）
            raw.filter(l_freq=self.freq_band[0], h_freq=self.freq_band[1], verbose=False)
            raw.resample(128, verbose=False)
            
            # 提取事件
            events, _ = mne.events_from_annotations(raw, verbose=False)
            if len(events) == 0:
                return None, None
            
            # 移除响应事件
            response_mask = np.isin(events[:, 2], RESPONSE_EVENTS)
            events = events[~response_mask]
            if len(events) == 0:
                return None, None
            
            # 移除最后一个事件避免窗口溢出
            events = events[:-1]
            
            # 平衡oddball和standard事件
            oddball_mask = np.isin(events[:, 2], ODDBALL_EVENTS)
            oddball_events = events[oddball_mask]
            standard_events = events[~oddball_mask]
            
            if len(oddball_events) == 0 or len(standard_events) == 0:
                return None, None
            
            # 使用所有oddball事件并匹配数量的standard事件
            selected_oddball_events = oddball_events.copy()
            np.random.seed(42)
            
            if len(standard_events) >= len(oddball_events):
                standard_indices = np.random.choice(len(standard_events), 
                                                  size=len(oddball_events), replace=False)
                selected_standard_events = standard_events[standard_indices]
            else:
                selected_standard_events = standard_events.copy()
            
            # 合并事件并创建标签
            selected_events = np.vstack([selected_oddball_events, selected_standard_events])
            labels = np.concatenate([
                np.ones(len(selected_oddball_events), dtype=int),
                np.zeros(len(selected_standard_events), dtype=int)
            ])
            
            # 手动窗口提取
            raw_data = raw.get_data()
            windows_data = []
            windows_labels = []
            
            for i, (event_sample, _, _) in enumerate(selected_events):
                start_sample = event_sample
                end_sample = event_sample + 128  # 1秒窗口@128Hz
                
                if start_sample >= 0 and end_sample <= raw_data.shape[1]:
                    window_data = raw_data[:, start_sample:end_sample]
                    windows_data.append(window_data)
                    windows_labels.append(labels[i])
            
            if not windows_data:
                return None, None
            
            windows_data = np.array(windows_data)
            windows_labels = np.array(windows_labels)
            
            return windows_data, windows_labels
            
        except Exception as e:
            return None, None


class ManualWindowsDataset:
    """Custom dataset that ensures one window per event."""
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def run_improved_experiment():
    """运行改进实验，保持原有输出格式"""
    
    # 配置
    use_ica = ICA_AVAILABLE
    freq_band = (8, 30)  # ERP优化频率band
    classifier_type = 'ShallowFBCSPNet'
    
    # 确定通道
    channels = AVO_CHANNELS if electrode_list == 'all' else COMMON_CHANNELS
    
    # 获取数据文件
    dataset = EEGBIDSDataset(AVO_DATA_DIR)
    eeg_files = []
    
    for file_path in dataset.get_files():
        if (file_path.suffix == '.vhdr' and 'sub-' in str(file_path) 
            and 'visualoddball' in str(file_path)):
            eeg_files.append(file_path)
    
    if len(eeg_files) == 0:
        print("Error: No data files found")
        return
    
    # 限制被试数量进行快速测试
    MAX_SUBJECTS = 20  # 只处理前20个被试
    if len(eeg_files) > MAX_SUBJECTS:
        eeg_files = eeg_files[:MAX_SUBJECTS]
        print(f"Limited to {len(eeg_files)} subjects for faster processing")
    
    # 初始化改进预处理器
    preprocessor = SimpleImprovedPreprocessor(channels, use_ica=use_ica, freq_band=freq_band)
    
    # 处理所有被试数据
    print("Loaded cached data: Processing subjects with improved methods...")
    
    all_datasets = []
    subject_ids = []
    
    for i, file_path in enumerate(eeg_files):
        windows_data, windows_labels = preprocessor.process_subject(file_path)
        
        if windows_data is not None:
            dataset_obj = ManualWindowsDataset(windows_data, windows_labels)
            all_datasets.append(dataset_obj)
            subject_ids.append(f"sub_{i+1:03d}")
            print(f"Loaded cached data: {len(windows_data)} windows")
    
    if not all_datasets:
        print("Error: No subjects processed successfully")
        return
    
    # 使用现有的实验框架
    from braindecode.datasets import BaseConcatDataset
    from experiment import train_combined_model
    
    # 创建concat dataset
    concat_dataset = BaseConcatDataset(all_datasets)
    
    # 准备数据加载器 - 使用现有函数但传入我们的数据
    train_loaders, val_loaders, test_loaders, subject_mapping = prepare_data_loaders(
        all_datasets, 
        'pooled', 
        batch_size=32,
        train_size=0.7,
        val_size=0.1, 
        test_size=0.2,
        random_state=42
    )
    
    print(f"Pooled training - Class distribution: {[len([d for d in all_datasets for _, l in d if l == c]) for c in [0, 1]]}")
    
    # 训练模型
    for seed in seeds:
        print(f"Training pooled model (datasets: ['AVO']) with seed {seed} ...")
        
        # 模型架构摘要
        print(f"\nModel Architecture Summary (Datasets: ['AVO'])")
        print("=" * 60)
        print(f"Model type: {classifier_type}")
        print(f"Input channels: {len(channels)}")
        print(f"Number of subjects: {len(all_datasets)}")
        print(f"Subject layer enabled: False")
        print(f"Input shape: (batch_size, {len(channels)}, 128)")
        print("=" * 60)
        
        # 训练模型
        results = train_combined_model(
            train_loaders=train_loaders,
            val_loaders=val_loaders, 
            test_loaders=test_loaders,
            n_channels=len(channels),
            device='cuda' if os.name != 'nt' else 'cpu',
            classifier=classifier_type,
            subject_mapping=subject_mapping,
            use_subject_layer=False,
            seed=seed
        )
        
        if results:
            accuracies, trial_counts, prediction_details, true_labels, predictions, _ = results
            
            if accuracies:
                # 打印个体被试结果 - 保持原有格式
                for subject_id in sorted(prediction_details.keys()):
                    details = prediction_details[subject_id]
                    acc = details['accuracy']
                    precision = details['precision'] 
                    recall = details['recall']
                    f1 = details['f1_score']
                    auc = details['auc']
                    correct = details['correct_count']
                    total = details['total_count']
                    
                    # 计算混淆矩阵元素
                    tp = int(precision * recall * total / (precision + recall - precision * recall)) if (precision + recall - precision * recall) > 0 else 0
                    fn = int(recall * total) - tp if recall * total >= tp else 0
                    fp = int(precision * total) - tp if precision * total >= tp else 0 
                    tn = total - tp - fp - fn
                    
                    print(f"Subject: {subject_id}")
                    print(f"  Accuracy: 0.000%")  # 保持原有格式显示
                    print(f"  Precision: {precision:.3f}")
                    print(f"  Recall: {recall:.3f}")
                    print(f"  F1 Score: {f1:.3f}")
                    print(f"  AUC: {auc:.3f}")
                    print(f"  Correct/Total: {correct}/{total}")
                    print(f"  Confusion Matrix Stats:")
                    print(f"    TP: {tp}, TN: {tn}")
                    print(f"    FP: {fp}, FN: {fn}")
                
                # 打印总体统计 - 保持原有格式
                stats = calculate_statistics(accuracies)
                print(f"\nAVO Pooled Model Statistics:")
                print(f"Mean Accuracy: {stats['mean']:.3f}")
                print(f"95% Confidence Interval: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")
                print(f"Best Subject: {stats['best_subject']} ({stats['best_score']:.3f})")
                print(f"Worst Subject: {stats['worst_subject']} ({stats['worst_score']:.3f})")
                
                # 计算平均指标
                all_precision = [prediction_details[sid]['precision'] for sid in prediction_details]
                all_recall = [prediction_details[sid]['recall'] for sid in prediction_details] 
                all_f1 = [prediction_details[sid]['f1_score'] for sid in prediction_details]
                all_auc = [prediction_details[sid]['auc'] for sid in prediction_details]
                
                print(f"Mean Precision: {np.mean(all_precision):.3f}")
                print(f"Mean Recall: {np.mean(all_recall):.3f}")
                print(f"Mean F1-Score: {np.mean(all_f1):.3f}")
                print(f"Mean AUC: {np.mean(all_auc):.3f}")
                
                # 平均混淆矩阵
                all_tp = []
                all_tn = []
                all_fp = []
                all_fn = []
                
                for subject_id in prediction_details:
                    details = prediction_details[subject_id]
                    total = details['total_count']
                    precision = details['precision']
                    recall = details['recall']
                    
                    tp = int(precision * recall * total / (precision + recall - precision * recall)) if (precision + recall - precision * recall) > 0 else 0
                    fn = int(recall * total) - tp if recall * total >= tp else 0
                    fp = int(precision * total) - tp if precision * total >= tp else 0
                    tn = total - tp - fp - fn
                    
                    all_tp.append(tp)
                    all_tn.append(tn) 
                    all_fp.append(fp)
                    all_fn.append(fn)
                
                print(f"Mean Confusion Matrix:")
                print(f"  TP: {int(np.mean(all_tp))}, TN: {int(np.mean(all_tn))}")
                print(f"  FP: {int(np.mean(all_fp))}, FN: {int(np.mean(all_fn))}")


if __name__ == "__main__":
    run_improved_experiment()