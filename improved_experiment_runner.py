"""
Improved experiment runner with advanced models and preprocessing
"""

import os
import sys
import torch
import numpy as np
import mne
import logging
import warnings
from pathlib import Path

# Import advanced components
from advanced_preprocessor import AdvancedOddballPreprocessor
from advanced_models import create_advanced_model
from advanced_augmentation import EEGAugmentation, AdvancedLoss

# Import existing components
from data_utils import EEGBIDSDataset
from constants import COMMON_CHANNELS, P3_CHANNELS, AVO_CHANNELS
from experiment import (
    prepare_data_loaders, train_model_with_loaders, 
    evaluate_model_with_loader
)
from utils import calculate_statistics, print_statistics, get_channel_list
from experiment_logger import (
    setup_logger, log_section_header, log_configuration, 
    log_individual_results, log_detailed_results, log_overall_metrics
)

# Import improved configuration
import config_improved as config

# Setup logging and warnings
mne.set_log_level('ERROR')
logging.getLogger('joblib').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


class ImprovedExperimentRunner:
    """Advanced experiment runner with improved models and preprocessing."""
    
    def __init__(self):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize advanced components
        self.augmentation = EEGAugmentation(**config.AUGMENTATION_CONFIG) if config.USE_ADVANCED_AUGMENTATION else None
        self.advanced_loss = AdvancedLoss(
            alpha=config.FOCAL_ALPHA,
            gamma=config.FOCAL_GAMMA,
            label_smoothing=config.LABEL_SMOOTHING,
            use_focal=config.USE_FOCAL_LOSS,
            use_label_smoothing=config.USE_LABEL_SMOOTHING
        )
    
    def get_preprocessor(self, channels):
        """Get appropriate preprocessor based on configuration."""
        if config.use_advanced_preprocessing:
            return AdvancedOddballPreprocessor(
                eeg_channels=channels,
                trial_start_offset_samples=config.TRIAL_START_OFFSET_SAMPLES,
                trial_stop_offset_samples=config.TRIAL_STOP_OFFSET_SAMPLES,
                use_ica=config.use_ica_artifact_removal,
                use_csp=config.use_csp_spatial_filtering,
                n_csp_components=config.n_csp_components,
                ica_n_components=config.ica_n_components,
                freq_band=config.ERP_FREQ_BAND,
                use_cache=config.USE_DATA_CACHE
            )
        else:
            # Fallback to original preprocessor
            from preprocessor import OddballPreprocessor
            return OddballPreprocessor(
                eeg_channels=channels,
                use_cache=config.USE_DATA_CACHE,
                fixed_trials_per_class=config.FIXED_TRIALS_PER_CLASS
            )
    
    def prepare_dataset(self, data_dir, channels, max_subjects=None):
        """Prepare dataset with advanced preprocessing."""
        print(f"Preparing dataset from: {data_dir}")
        print(f"Using channels: {channels[:5]}... (showing first 5)")
        
        # Get EEG files
        dataset = EEGBIDSDataset(data_dir)
        eeg_files = []
        
        for file_path in dataset.get_files():
            if file_path.suffix == '.edf' and 'sub-' in str(file_path):
                eeg_files.append(file_path)
        
        if not eeg_files:
            raise ValueError(f"No EEG files found in {data_dir}")
        
        # Limit subjects for quick testing
        if max_subjects and len(eeg_files) > max_subjects:
            eeg_files = eeg_files[:max_subjects]
            print(f"Limited to {len(eeg_files)} subjects for quick testing")
        
        # Get preprocessor
        preprocessor = self.get_preprocessor(channels)
        
        # Process each file
        all_windows = []
        all_labels = []
        all_subject_ids = []
        
        for i, file_path in enumerate(eeg_files):
            try:
                print(f"Processing {i+1}/{len(eeg_files)}: {file_path.name}")
                
                # Load raw data
                raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
                
                # Apply preprocessing
                windowed_dataset = preprocessor.transform(raw)
                
                # Extract data and labels
                windows = windowed_dataset.data
                labels = windowed_dataset.labels
                
                # Store with subject IDs
                all_windows.extend(windows)
                all_labels.extend(labels)
                all_subject_ids.extend([i] * len(windows))
                
                print(f"  Extracted {len(windows)} windows")
                
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
                continue
        
        if not all_windows:
            raise ValueError("No valid data extracted from any files")
        
        # Convert to arrays
        X = np.array(all_windows)
        y = np.array(all_labels)
        subject_ids = np.array(all_subject_ids)
        
        print(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} channels, {X.shape[2]} timepoints")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y, subject_ids
    
    def create_model(self, n_channels, model_type=None):
        """Create model based on configuration."""
        if model_type is None:
            model_type = config.classifier
        
        print(f"Creating {model_type} model with {n_channels} channels")
        
        if model_type == 'lda':
            return create_advanced_model('lda', n_channels)
        else:
            model = create_advanced_model(model_type, n_channels)
            model = model.to(self.device)
            return model
    
    def train_advanced_model(self, model, train_loader, val_loader, test_loader):
        """Train model with advanced techniques."""
        if config.classifier == 'lda':
            # LDA training (existing implementation)
            from models import train_model
            return train_model(model, train_loader, val_loader, test_loader, 
                             self.device, is_lda=True)
        
        # Neural network training with advanced features
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Advanced scheduler
        if config.USE_COSINE_ANNEALING:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.COSINE_T_MAX
            )
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.GAMMA)
        
        # Warmup scheduler
        warmup_scheduler = None
        if config.USE_WARMUP:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=config.WARMUP_EPOCHS
            )
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(config.MAX_EPOCHS):
            model.train()
            total_loss = 0
            
            for batch_data in train_loader:
                if len(batch_data) == 3:
                    x, y, subject_indices = batch_data
                    subject_indices = subject_indices.to(self.device)
                else:
                    x, y = batch_data
                    subject_indices = None
                
                # Apply advanced augmentation
                if self.augmentation is not None:
                    x, y = self.augmentation(x.to(self.device), y.to(self.device), 
                                           training=True, apply_mixup=True)
                else:
                    x, y = x.to(self.device), y.to(self.device)
                
                # Normalize data
                from models import normalize_data
                x = normalize_data(x)
                
                if y.ndim > 1:
                    y = torch.argmax(y, dim=1)
                
                optimizer.zero_grad()
                
                # Forward pass
                if hasattr(model, 'subject_layer') and subject_indices is not None:
                    outputs = model(x, subject_indices)
                else:
                    outputs = model(x)
                
                # Compute loss
                loss = self.advanced_loss(outputs, y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Update schedulers
            if warmup_scheduler is not None and epoch < config.WARMUP_EPOCHS:
                warmup_scheduler.step()
            else:
                scheduler.step()
            
            # Validation
            val_acc = self.evaluate_model(model, val_loader)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if config.VERBOSE_TRAINING and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}, "
                      f"Val Acc = {val_acc:.4f}")
            
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model and evaluate
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return self.evaluate_model(model, test_loader, return_details=True)
    
    def evaluate_model(self, model, data_loader, return_details=False):
        """Evaluate model with detailed metrics."""
        if config.classifier == 'lda':
            from models import evaluate
            return evaluate(model, data_loader, self.device, is_lda=True, 
                          return_details=return_details)
        
        # Neural network evaluation
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                if len(batch_data) == 3:
                    x, y, subject_indices = batch_data
                    subject_indices = subject_indices.to(self.device)
                else:
                    x, y = batch_data
                    subject_indices = None
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Normalize data
                from models import normalize_data
                x = normalize_data(x)
                
                if y.ndim > 1:
                    y = torch.argmax(y, dim=1)
                
                # Forward pass
                if hasattr(model, 'subject_layer') and subject_indices is not None:
                    outputs = model(x, subject_indices)
                else:
                    outputs = model(x)
                
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                if return_details:
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
        
        accuracy = correct / total
        
        if return_details:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            
            precision = precision_score(all_targets, all_predictions, average='binary', zero_division=0)
            recall = recall_score(all_targets, all_predictions, average='binary', zero_division=0)
            f1 = f1_score(all_targets, all_predictions, average='binary', zero_division=0)
            
            try:
                auc = roc_auc_score(all_targets, all_predictions)
            except:
                auc = 0.5
            
            return {
                'accuracy': accuracy,
                'correct_count': correct,
                'incorrect_count': total - correct,
                'total_count': total,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }
        
        return accuracy
    
    def run_single_experiment(self, model_type=None, max_subjects=None):
        """Run a single experiment with specified configuration."""
        if model_type is None:
            model_type = config.classifier
        
        if max_subjects is None and config.QUICK_TEST_MODE:
            max_subjects = config.QUICK_TEST_SUBJECTS
        
        print(f"\n{'='*50}")
        print(f"Running Experiment: {model_type}")
        print(f"Advanced Preprocessing: {config.use_advanced_preprocessing}")
        print(f"Max Subjects: {max_subjects}")
        print(f"{'='*50}")
        
        # Determine channels
        if config.electrode_list == 'common':
            channels = COMMON_CHANNELS
        elif 'ds005863' in config.dataset:
            channels = AVO_CHANNELS if config.electrode_list == 'all' else COMMON_CHANNELS
        else:
            channels = P3_CHANNELS if config.electrode_list == 'all' else COMMON_CHANNELS
        
        # Prepare dataset
        X, y, subject_ids = self.prepare_dataset(config.data_dir, channels, max_subjects)
        
        # Create data loaders
        from sklearn.model_selection import train_test_split
        from torch.utils.data import TensorDataset, DataLoader
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(config.VAL_SIZE + config.TEST_SIZE), 
            random_state=42, stratify=y
        )
        
        val_size_adjusted = config.VAL_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_size_adjusted),
            random_state=42, stratify=y_temp
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_val = torch.LongTensor(y_val)
        y_test = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        print(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Create and train model
        model = self.create_model(X.shape[1], model_type)
        results = self.train_advanced_model(model, train_loader, val_loader, test_loader)
        
        return results
    
    def run_model_comparison(self, max_subjects=None):
        """Run comparison of multiple models."""
        if not config.COMPARE_MODELS:
            return self.run_single_experiment(max_subjects=max_subjects)
        
        results = {}
        
        print(f"\n{'='*60}")
        print("Running Model Comparison")
        print(f"Models: {config.MODELS_TO_COMPARE}")
        print(f"{'='*60}")
        
        for model_type in config.MODELS_TO_COMPARE:
            try:
                print(f"\n--- Testing {model_type} ---")
                result = self.run_single_experiment(model_type, max_subjects)
                results[model_type] = result
                
                if isinstance(result, dict):
                    print(f"{model_type} Results:")
                    print(f"  Accuracy: {result['accuracy']:.4f}")
                    print(f"  F1 Score: {result['f1_score']:.4f}")
                    print(f"  AUC: {result['auc']:.4f}")
                else:
                    print(f"{model_type} Accuracy: {result:.4f}")
                    
            except Exception as e:
                print(f"Error with {model_type}: {e}")
                results[model_type] = None
        
        # Print comparison summary
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        for model_type, result in results.items():
            if result is not None:
                if isinstance(result, dict):
                    acc = result['accuracy']
                    f1 = result['f1_score']
                    auc = result['auc']
                    print(f"{model_type:20s}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
                else:
                    print(f"{model_type:20s}: Accuracy={result:.4f}")
            else:
                print(f"{model_type:20s}: FAILED")
        
        return results


def main():
    """Main experiment runner."""
    try:
        # Create experiment runner
        runner = ImprovedExperimentRunner()
        
        # Run experiments
        results = runner.run_model_comparison()
        
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        return results
        
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()