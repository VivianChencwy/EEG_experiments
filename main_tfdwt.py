"""
Target-Focused Domain Weighted Training (TF-DWT)

This standalone entry point trains with full AVO and few-shot P3 jointly,
without changing the backbone architecture or parameter count.

Key features:
- Uses per-subject trial budgets from config (NESTED_CV_TRIALS_PER_SUBJECT_*).
- Automatically detects domain imbalance and schedules:
  - P3 loss weight w_P3 = clip(sqrt(N_AVO/N_P3), [1.5, 6.0]) with warmup.
  - MMD alignment weight λ with warmup (no new parameters).
- Split-BN buffers (gamma/beta shared): maintains per-domain running stats
  for BatchNorm layers during training, and uses P3 stats for P3 eval.
- Detailed logging mirroring main.py sections plus adjustment monitoring.
- Early stopping based on P3 validation performance with guardrails.

Run: python main_tfdwt.py
"""

import os
import math
import logging
import warnings
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import mne

from config import (
    P3_DATA_DIR, AVO_DATA_DIR,
    NESTED_CV_TRIALS_PER_SUBJECT_P3, NESTED_CV_TRIALS_PER_SUBJECT_AVO,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
    BATCH_SIZE, MAX_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE, EARLY_STOPPING_PATIENCE,
    seeds,
    use_combined_datasets, separate_subject_classification, electrode_list,
    ELECTRODE_FUSION_METHOD, DOMAIN_ADAPTATION_METHOD,
    USE_ENHANCED_PREPROCESSING,
    USE_DATA_AUGMENTATION, NOISE_STD, TIME_SHIFT_RANGE, LABEL_SMOOTHING,
    DEVICE_MODE,
)
from constants import COMMON_CHANNELS, P3_CHANNELS, AVO_CHANNELS
from experiment import (
    get_device,
    get_dataset_subjects,
    create_preprocessor,
    stratified_sample_trials,
)
from utils import process_subject_data
from models import create_model, normalize_data
from experiment_logger import (
    setup_logger, log_section_header, log_configuration, cleanup_failed_log
)


# ---- Utilities ----

def set_global_torch_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, logger: logging.Logger, eps: float = 1e-8) -> torch.Tensor:
    """Compute unbiased RBF-MMD between two batches (features or logits).
    No trainable parameters; kernel bandwidth via median heuristic.
    """
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    if y.dim() > 2:
        y = y.view(y.size(0), -1)
    with torch.no_grad():
        # Median heuristic on combined data
        z = torch.cat([x, y], dim=0)
        if z.size(0) > 1:
            dists = torch.cdist(z, z, p=2.0)
            sigma = torch.median(dists)
            sigma = torch.clamp(sigma, min=eps)
        else:
            sigma = torch.tensor(1.0, device=z.device)
    gamma = 1.0 / (2.0 * (sigma ** 2) + eps)
    k_xx = torch.exp(-gamma * torch.cdist(x, x, p=2.0) ** 2)
    k_yy = torch.exp(-gamma * torch.cdist(y, y, p=2.0) ** 2)
    k_xy = torch.exp(-gamma * torch.cdist(x, y, p=2.0) ** 2)
    m = x.size(0)
    n = y.size(0)
    if m <= 1 or n <= 1:
        return torch.tensor(0.0, device=x.device)
    # Unbiased estimate: exclude diagonals
    mmd = (k_xx.sum() - torch.trace(k_xx)) / (m * (m - 1) + eps)
    mmd += (k_yy.sum() - torch.trace(k_yy)) / (n * (n - 1) + eps)
    mmd -= 2.0 * k_xy.mean()
    return mmd


def snapshot_bn_buffers(model: nn.Module) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Capture running_mean and running_var tensors of all BN layers."""
    buffers = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Clone to detach current buffers
            rm = m.running_mean.clone() if m.running_mean is not None else None
            rv = m.running_var.clone() if m.running_var is not None else None
            buffers.append((rm, rv))
    return buffers


def restore_bn_buffers(model: nn.Module, buffers: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Restore running_mean and running_var of BN layers from snapshot."""
    idx = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            rm, rv = buffers[idx]
            if rm is not None and m.running_mean is not None:
                m.running_mean.data.copy_(rm)
            if rv is not None and m.running_var is not None:
                m.running_var.data.copy_(rv)
            idx += 1


def get_channels_for_dataset(name: str, use_all: bool) -> List[str]:
    if name == 'P3':
        return P3_CHANNELS if use_all else COMMON_CHANNELS
    elif name == 'AVO':
        return AVO_CHANNELS if use_all else COMMON_CHANNELS
    else:
        return COMMON_CHANNELS


def prepare_domain_arrays(logger: logging.Logger, channels: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Load per-domain arrays with per-subject stratified sampling from config.
    Returns dict: { 'P3': {X_train,...}, 'AVO': {X_train,...} }
    """
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for dataset_name in ['P3', 'AVO']:
        logger.info(f"Preparing dataset: {dataset_name}")
        # Subject listing
        if dataset_name == 'P3':
            subjects = get_dataset_subjects('P3', P3_DATA_DIR)
            dataset_obj = P3_DATA_DIR
            n_trials_ps = NESTED_CV_TRIALS_PER_SUBJECT_P3
        else:
            from data_utils import EEGBIDSDataset
            avo_dataset = EEGBIDSDataset(data_dir=AVO_DATA_DIR, dataset='ds005863')
            subjects = get_dataset_subjects('AVO', avo_dataset)
            dataset_obj = avo_dataset
            n_trials_ps = NESTED_CV_TRIALS_PER_SUBJECT_AVO

        preproc = create_preprocessor(channels, dataset_name)

        all_X = []
        all_y = []
        for s in subjects:
            data, labels = process_subject_data(s, dataset_obj, preproc, logger, dataset_type=dataset_name)
            if data is None or labels is None or len(data) == 0:
                continue
            # Standardize labels
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.squeeze()
            # Stratified per-subject sampling
            if len(data) > n_trials_ps:
                data, labels = stratified_sample_trials(data, labels, n_trials_ps, f"{dataset_name}_{s}", logger)
            all_X.append(data)
            all_y.append(labels)

        if not all_X:
            raise RuntimeError(f"No valid data for {dataset_name}")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        # Split by ratios (stratified)
        temp_size = VAL_SIZE + TEST_SIZE
        idx_all = np.arange(len(X))
        train_idx, temp_idx = train_test_split(idx_all, test_size=temp_size, stratify=y, random_state=42)
        test_ratio = TEST_SIZE / temp_size if temp_size > 0 else 0.5
        val_idx, test_idx = train_test_split(temp_idx, test_size=test_ratio, stratify=y[temp_idx], random_state=42) if len(temp_idx) > 0 else (np.array([], dtype=int), np.array([], dtype=int))

        results[dataset_name] = {
            'X_train': X[train_idx], 'y_train': y[train_idx],
            'X_val': X[val_idx], 'y_val': y[val_idx],
            'X_test': X[test_idx], 'y_test': y[test_idx],
        }

        logger.info(f"{dataset_name} data prepared: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return results


def get_adjustments(n_train_avo: int, n_train_p3: int) -> Tuple[float, float, int]:
    """Compute automatic weights based on imbalance.
    Returns (w_p3_target, lambda_mmd_target, warmup_epochs)
    """
    # Avoid division by zero
    n_train_p3 = max(1, n_train_p3)
    ratio = n_train_avo / float(n_train_p3)
    w_p3 = math.sqrt(ratio)
    w_p3 = float(np.clip(w_p3, 1.5, 6.0))
    # MMD a bit stronger with larger shift
    lambda_mmd = 0.1 if ratio < 2.0 else (0.2 if ratio < 4.0 else 0.3)
    warmup = max(2, min(5, int(0.1 * MAX_EPOCHS)))
    return w_p3, lambda_mmd, warmup


def evaluate_domain(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = normalize_data(x).to(device)
            y = y.to(device)
            scores = model(x)
            if scores.ndim > 2:
                scores = scores.view(scores.size(0), -1)
            _, pred = scores.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def tfdwt_train(logger: logging.Logger, datasets_dict: Dict[str, Dict[str, np.ndarray]], channels: List[str], seed: int = 42) -> Dict[str, float]:
    device = get_device()
    set_global_torch_seed(seed)

    # Build loaders per domain
    Xtr_p3 = torch.FloatTensor(datasets_dict['P3']['X_train'])
    ytr_p3 = torch.LongTensor(datasets_dict['P3']['y_train'])
    Xva_p3 = torch.FloatTensor(datasets_dict['P3']['X_val'])
    yva_p3 = torch.LongTensor(datasets_dict['P3']['y_val'])
    Xte_p3 = torch.FloatTensor(datasets_dict['P3']['X_test'])
    yte_p3 = torch.LongTensor(datasets_dict['P3']['y_test'])

    Xtr_avo = torch.FloatTensor(datasets_dict['AVO']['X_train'])
    ytr_avo = torch.LongTensor(datasets_dict['AVO']['y_train'])
    Xva_avo = torch.FloatTensor(datasets_dict['AVO']['X_val'])
    yva_avo = torch.LongTensor(datasets_dict['AVO']['y_val'])
    Xte_avo = torch.FloatTensor(datasets_dict['AVO']['X_test'])
    yte_avo = torch.LongTensor(datasets_dict['AVO']['y_test'])

    train_loader_p3 = DataLoader(TensorDataset(Xtr_p3, ytr_p3), batch_size=BATCH_SIZE, shuffle=True)
    val_loader_p3 = DataLoader(TensorDataset(Xva_p3, yva_p3), batch_size=BATCH_SIZE, shuffle=False)
    test_loader_p3 = DataLoader(TensorDataset(Xte_p3, yte_p3), batch_size=BATCH_SIZE, shuffle=False)

    train_loader_avo = DataLoader(TensorDataset(Xtr_avo, ytr_avo), batch_size=BATCH_SIZE, shuffle=True)
    val_loader_avo = DataLoader(TensorDataset(Xva_avo, yva_avo), batch_size=BATCH_SIZE, shuffle=False)
    test_loader_avo = DataLoader(TensorDataset(Xte_avo, yte_avo), batch_size=BATCH_SIZE, shuffle=False)

    # Create model with actual input channels (enhanced preprocessing may expand channels)
    input_channels = Xtr_avo.shape[1] if Xtr_avo.shape[1] == Xtr_p3.shape[1] else max(Xtr_avo.shape[1], Xtr_p3.shape[1])
    model = create_model(n_channels=len(channels), is_lda=False, input_channels=input_channels)
    model = model.to(device)

    # Optimizer & schedule
    optimizer = torch.optim.Adamax(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # Automatic adjustments based on imbalance
    n_train_avo = len(Xtr_avo)
    n_train_p3 = len(Xtr_p3)
    w_p3_target, lambda_mmd_target, warmup_epochs = get_adjustments(n_train_avo, n_train_p3)
    logger.info(f"Auto adjustments: train_AVO={n_train_avo}, train_P3={n_train_p3}, w_P3_target={w_p3_target:.3f}, lambda_MMD_target={lambda_mmd_target:.3f}, warmup_epochs={warmup_epochs}")

    # Early stopping (P3-val)
    best_p3_val = 0.0
    best_state = None
    patience_cnt = 0

    # Split-BN buffers for two domains (gamma/beta shared)
    bn_store: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {
        'AVO': snapshot_bn_buffers(model),
        'P3': snapshot_bn_buffers(model),
    }

    # Monitoring guards
    def guard_adjustments(p3_val_hist: List[float], cur_w: float, cur_lambda: float) -> Tuple[float, float]:
        if len(p3_val_hist) >= 3 and p3_val_hist[-1] < p3_val_hist[-2] < p3_val_hist[-3]:
            # 3 consecutive drops on P3 val -> back off
            new_w = max(1.5, cur_w * 0.8)
            new_lambda = max(0.0, cur_lambda * 0.5)
            logger.warning(f"P3 val decreasing 3x in a row. Reducing w_P3 to {new_w:.3f}, lambda_MMD to {new_lambda:.3f}")
            return new_w, new_lambda
        return cur_w, cur_lambda

    p3_val_hist: List[float] = []

    # Training loop
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        # Warmup schedules
        alpha = min(1.0, epoch / max(1, warmup_epochs))
        w_p3 = 1.0 + alpha * (w_p3_target - 1.0)
        lambda_mmd = alpha * lambda_mmd_target

        # Optionally apply guards
        w_p3, lambda_mmd = guard_adjustments(p3_val_hist, w_p3, lambda_mmd)

        # Log adjustments
        lr_cur = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}/{MAX_EPOCHS} | LR={lr_cur:.6f} | w_P3={w_p3:.3f} | lambda_MMD={lambda_mmd:.3f}")

        # Iterators
        itr_avo = iter(train_loader_avo)
        itr_p3 = iter(train_loader_p3) if len(train_loader_p3) > 0 else None

        steps = 0
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        while True:
            try:
                xb_avo, yb_avo = next(itr_avo)
            except StopIteration:
                break  # finished full AVO pass (we do not drop AVO samples)

            # Try get a P3 batch (with replacement if exhausted)
            if itr_p3 is None:
                xb_p3 = None
                yb_p3 = None
            else:
                try:
                    xb_p3, yb_p3 = next(itr_p3)
                except StopIteration:
                    # restart P3 iterator (oversampling)
                    itr_p3 = iter(train_loader_p3)
                    xb_p3, yb_p3 = next(itr_p3) if len(train_loader_p3) > 0 else (None, None)

            optimizer.zero_grad()

            # Domain AVO forward (set BN buffers for AVO)
            restore_bn_buffers(model, bn_store['AVO'])
            x_avo = normalize_data(xb_avo).to(device)
            y_avo = yb_avo.to(device)
            scores_avo = model(x_avo)
            if scores_avo.ndim > 2:
                scores_avo = scores_avo.view(scores_avo.size(0), -1)
            loss_avo = F.cross_entropy(scores_avo, y_avo)
            # Capture updated AVO BN buffers
            bn_store['AVO'] = snapshot_bn_buffers(model)

            # Domain P3 forward (if available)
            loss_p3 = torch.tensor(0.0, device=device)
            scores_p3 = None
            if xb_p3 is not None:
                restore_bn_buffers(model, bn_store['P3'])
                x_p3 = normalize_data(xb_p3).to(device)
                y_p3 = yb_p3.to(device)
                scores_p3 = model(x_p3)
                if scores_p3.ndim > 2:
                    scores_p3 = scores_p3.view(scores_p3.size(0), -1)
                loss_p3 = F.cross_entropy(scores_p3, y_p3)
                bn_store['P3'] = snapshot_bn_buffers(model)

            # Alignment loss on logits (no new params)
            loss_align = torch.tensor(0.0, device=device)
            if (scores_p3 is not None) and (lambda_mmd > 0.0):
                try:
                    # Use min batch size for stability
                    b = min(scores_avo.size(0), scores_p3.size(0))
                    loss_align = compute_mmd_rbf(scores_avo[:b].detach(), scores_p3[:b].detach(), logger)
                except Exception as e:
                    logger.warning(f"MMD computation failed: {e}; skipping this step")
                    loss_align = torch.tensor(0.0, device=device)

            total_loss = loss_avo + w_p3 * loss_p3 + lambda_mmd * loss_align
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning("Encountered NaN/Inf loss; skipping step and reducing LR by 2x")
                for pg in optimizer.param_groups:
                    pg['lr'] = max(1e-6, pg['lr'] * 0.5)
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Basic training stats (on AVO batch)
            with torch.no_grad():
                _, pred = scores_avo.max(1)
                epoch_correct += (pred == y_avo).sum().item()
                epoch_total += y_avo.size(0)
                epoch_loss += total_loss.item()
            steps += 1

        # End epoch
        scheduler.step()
        avg_tr_loss = epoch_loss / max(1, steps)
        tr_acc = epoch_correct / max(1, epoch_total)
        p3_val = evaluate_domain(model, val_loader_p3, device)
        avo_val = evaluate_domain(model, val_loader_avo, device)
        p3_val_hist.append(p3_val)

        logger.info(f"Epoch Summary: TrainLoss={avg_tr_loss:.4f} | TrainAcc={tr_acc:.3f} | Val(P3)={p3_val:.3f} | Val(AVO)={avo_val:.3f}")

        # Early stopping on P3 val
        improved = p3_val > best_p3_val + 1e-4
        if improved:
            best_p3_val = p3_val
            best_state = model.state_dict()
            patience_cnt = 0
            logger.info("New best P3-val; checkpoint updated")
        else:
            patience_cnt += 1
            remaining = max(0, EARLY_STOPPING_PATIENCE - patience_cnt)
            logger.info(f"No P3-val improvement; patience remaining: {remaining}/{EARLY_STOPPING_PATIENCE}")
            if patience_cnt >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered (P3-val)")
                break

    # Load best & evaluate
    if best_state is not None:
        model.load_state_dict(best_state)

    p3_test_acc = evaluate_domain(model, test_loader_p3, device)
    avo_test_acc = evaluate_domain(model, test_loader_avo, device)
    overall = (p3_test_acc + avo_test_acc) / 2.0 if (p3_test_acc > 0 and avo_test_acc > 0) else max(p3_test_acc, avo_test_acc)

    logger.info(f"Test Accuracy - P3: {p3_test_acc:.4f}, AVO: {avo_test_acc:.4f}, Overall: {overall:.4f}")

    return {
        'p3_val_best': best_p3_val,
        'p3_test': p3_test_acc,
        'avo_test': avo_test_acc,
        'overall_test': overall,
    }


def main():
    mne.set_log_level('ERROR')
    logging.getLogger('joblib').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')

    logger = None
    try:
        # Determine channels per combined-training rule (no fusion/domain methods => use COMMON)
        if ELECTRODE_FUSION_METHOD == 'none' and DOMAIN_ADAPTATION_METHOD == 'none':
            if electrode_list != 'common':
                print("Warning: Combined training without fusion/domain methods uses COMMON channels")
            channels = COMMON_CHANNELS
        else:
            # Not expected per user instruction; still fallback to COMMON
            channels = COMMON_CHANNELS

        # Logger setup
        logger = setup_logger('TF_DWT', create_file=True)
        log_section_header(logger, "TF-DWT Joint Training (Auto-Adjusted)")

        # Log configuration summary
        log_configuration(logger, {
            'electrode_list': electrode_list,
            'fusion_method': ELECTRODE_FUSION_METHOD,
            'domain_adaptation': DOMAIN_ADAPTATION_METHOD,
            'use_enhanced_preprocessing': USE_ENHANCED_PREPROCESSING,
            'batch_size': BATCH_SIZE,
            'max_epochs': MAX_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'dropout_rate': DROPOUT_RATE,
            'use_data_augmentation': USE_DATA_AUGMENTATION,
            'noise_std': NOISE_STD,
            'time_shift_range': TIME_SHIFT_RANGE,
            'label_smoothing': LABEL_SMOOTHING,
            'trials_per_subject_P3': NESTED_CV_TRIALS_PER_SUBJECT_P3,
            'trials_per_subject_AVO': NESTED_CV_TRIALS_PER_SUBJECT_AVO,
            'train/val/test': (TRAIN_SIZE, VAL_SIZE, TEST_SIZE),
            'device_mode': DEVICE_MODE,
        })

        # Prepare data
        log_section_header(logger, "Preparing Data (Per-Subject Stratified Sampling)")
        datasets_dict = prepare_domain_arrays(logger, channels)

        # Auto-imbalance summary
        n_tr_avo = len(datasets_dict['AVO']['X_train'])
        n_tr_p3 = len(datasets_dict['P3']['X_train'])
        n_va_avo = len(datasets_dict['AVO']['X_val'])
        n_va_p3 = len(datasets_dict['P3']['X_val'])
        n_te_avo = len(datasets_dict['AVO']['X_test'])
        n_te_p3 = len(datasets_dict['P3']['X_test'])
        logger.info(f"Train sizes | AVO={n_tr_avo}, P3={n_tr_p3}")
        logger.info(f"Val sizes   | AVO={n_va_avo}, P3={n_va_p3}")
        logger.info(f"Test sizes  | AVO={n_te_avo}, P3={n_te_p3}")

        # Train with TF-DWT
        log_section_header(logger, "Training with TF-DWT (Auto Weights/Alignment/BN Stats)")
        results = tfdwt_train(logger, datasets_dict, channels, seed=seeds[0])

        # Report in a style similar to main.py
        log_section_header(logger, "CROSS-VALIDATION RESULTS (Proxy)")
        logger.info(f"P3 (proxy test) Accuracy: {results['p3_test']:.4f}")
        logger.info(f"AVO (proxy test) Accuracy: {results['avo_test']:.4f}")
        logger.info(f"Overall (proxy) Accuracy: {results['overall_test']:.4f}")

        print("\n--- Experiment Run Complete (TF-DWT) ---")

    except Exception as e:
        print(f"\n--- TF-DWT Experiment Failed: {e} ---")
        if logger:
            cleanup_failed_log(logger)
        raise
    except KeyboardInterrupt:
        print("\n--- TF-DWT Experiment Interrupted by User ---")
        if logger:
            cleanup_failed_log(logger)
        raise


if __name__ == "__main__":
    main()


