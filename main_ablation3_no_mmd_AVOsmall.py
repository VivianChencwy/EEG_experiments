"""
Ablation Study 3: Remove RBF-MMD Alignment

This version removes the MMD alignment loss completely.
Only domain-weighted cross-entropy losses are used.

Key modifications from TF-DWT:
- Remove MMD alignment (lambda_MMD = 0)
- Keep domain weighting with warmup
- Keep Split-BN
- Keep other training mechanisms
"""

import os
import sys
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
from scipy import stats

import mne

from config import (
    P3_DATA_DIR, AVO_DATA_DIR,
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
from models import create_model, normalize_data, evaluate
from experiment_logger import (
    setup_logger, log_section_header, log_configuration, cleanup_failed_log
)


# ---- Utilities ----

def set_global_torch_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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


def load_combined_arrays(logger: logging.Logger, channels: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load combined arrays (P3+AVO) with per-subject stratified sampling from config.
    Returns (X_all, y_all, domain_sources) where domain_sources is array of 'P3'/'AVO'.
    """
    X_list = []
    y_list = []
    src_list = []

    for dataset_name in ['P3', 'AVO']:
        logger.info(f"Loading dataset for CV: {dataset_name}")
        # Get dynamic config values
        import importlib.util
        import os

        # Force fresh import by loading config from batch_runner override path
        # This will load the temporary config.py created by batch_runner
        config_path = os.environ.get('CONFIG_OVERRIDE_PATH', os.path.join(os.getcwd(), 'config.py'))
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        if dataset_name == 'P3':
            subjects = get_dataset_subjects('P3', P3_DATA_DIR)
            dataset_obj = P3_DATA_DIR
            n_trials_ps = config.NESTED_CV_TRIALS_PER_SUBJECT_P3
        else:
            from data_utils import EEGBIDSDataset
            avo_dataset = EEGBIDSDataset(data_dir=AVO_DATA_DIR, dataset='ds005863')
            subjects = get_dataset_subjects('AVO', avo_dataset)
            dataset_obj = avo_dataset
            n_trials_ps = config.NESTED_CV_TRIALS_PER_SUBJECT_AVO

        preproc = create_preprocessor(channels, dataset_name)

        for s in subjects:
            data, labels = process_subject_data(s, dataset_obj, preproc, logger, dataset_type=dataset_name)
            if data is None or labels is None or len(data) == 0:
                continue
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.squeeze()
            # Per-subject stratified sampling to target budget
            if len(data) > n_trials_ps:
                data, labels = stratified_sample_trials(data, labels, n_trials_ps, f"{dataset_name}_{s}", logger)
            X_list.append(data)
            y_list.append(labels)
            src_list.append(np.array([dataset_name] * len(labels)))

    if not X_list:
        raise RuntimeError("No valid data loaded for CV")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    src_all = np.concatenate(src_list, axis=0)

    logger.info(f"Combined dataset summary: total={len(X_all)}, P3={np.sum(src_all=='P3')}, AVO={np.sum(src_all=='AVO')}")
    return X_all, y_all, src_all


def get_symmetric_adjustments(n_train_a: int, n_train_b: int) -> Tuple[float, int]:
    """Compute symmetric domain weights based purely on relative sizes.
    Returns (w_small_target, warmup_epochs).
    NOTE: No lambda_mmd in this ablation.
    """
    n_train_a = max(1, n_train_a)
    n_train_b = max(1, n_train_b)
    ratio_ab = n_train_a / float(n_train_b)
    # Representative target for logging; per-domain weights computed inside the loop
    w_small = float(np.clip(math.sqrt(1.0 / max(ratio_ab, 1.0)), 1.0, 6.0))
    warmup = max(2, min(5, int(0.1 * MAX_EPOCHS)))
    return w_small, warmup


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


def ablation3_train_fold(
    logger: logging.Logger,
    Xtr_p3: np.ndarray, ytr_p3: np.ndarray,
    Xva_p3: np.ndarray, yva_p3: np.ndarray,
    Xtr_avo: np.ndarray, ytr_avo: np.ndarray,
    Xva_avo: np.ndarray, yva_avo: np.ndarray,
    channels: List[str], seed: int = 42,
) -> nn.Module:
    """Ablation 3: Remove RBF-MMD alignment (lambda_MMD = 0)"""
    device = get_device()
    set_global_torch_seed(seed)

    # Build loaders per domain
    Xtr_p3 = torch.FloatTensor(Xtr_p3)
    ytr_p3 = torch.LongTensor(ytr_p3)
    Xva_p3 = torch.FloatTensor(Xva_p3)
    yva_p3 = torch.LongTensor(yva_p3)
    Xtr_avo = torch.FloatTensor(Xtr_avo)
    ytr_avo = torch.LongTensor(ytr_avo)
    Xva_avo = torch.FloatTensor(Xva_avo)
    yva_avo = torch.LongTensor(yva_avo)

    train_loader_p3 = DataLoader(TensorDataset(Xtr_p3, ytr_p3), batch_size=BATCH_SIZE, shuffle=True)
    val_loader_p3 = DataLoader(TensorDataset(Xva_p3, yva_p3), batch_size=BATCH_SIZE, shuffle=False)

    train_loader_avo = DataLoader(TensorDataset(Xtr_avo, ytr_avo), batch_size=BATCH_SIZE, shuffle=True)
    val_loader_avo = DataLoader(TensorDataset(Xva_avo, yva_avo), batch_size=BATCH_SIZE, shuffle=False)

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
    w_small, warmup_epochs = get_symmetric_adjustments(n_train_avo, n_train_p3)
    # Identify smaller domain for this fold
    small_domain = 'P3' if n_train_p3 <= n_train_avo else 'AVO'
    large_domain = 'AVO' if small_domain == 'P3' else 'P3'
    logger.info(f"Fold domains: small={small_domain} (n={min(n_train_p3, n_train_avo)}), large={large_domain} (n={max(n_train_p3, n_train_avo)})")
    logger.info(f"ABLATION 3: No MMD alignment (lambda_MMD=0), w_small≈{w_small:.3f}, warmup_epochs={warmup_epochs}")

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
    def guard_adjustments_small(val_history_small: List[float], cur_w: float, domain_name: str) -> float:
        if len(val_history_small) >= 3 and val_history_small[-1] < val_history_small[-2] < val_history_small[-3]:
            # 3 consecutive drops on small-domain val -> back off weight
            new_w = max(1.0, cur_w * 0.8)
            logger.warning(f"{domain_name} val decreasing 3x. Reducing w_small to {new_w:.3f}")
            return new_w
        return cur_w

    val_hist: Dict[str, List[float]] = {'P3': [], 'AVO': []}

    # Training loop
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        # Warmup schedules (symmetric)
        alpha = min(1.0, epoch / max(1, warmup_epochs))
        n_small = n_train_p3 if small_domain == 'P3' else n_train_avo
        n_large = n_train_avo if small_domain == 'P3' else n_train_p3
        # Emphasize the smaller domain; keep large at 1.0 (symmetric rule)
        w_small_target = max(1.0, math.sqrt(max(1, n_large) / max(1, n_small)))
        # Cap the max emphasis ratio to avoid suppressing large domain too much
        w_small_target = min(w_small_target, 3.0)
        w_large_target = 1.0
        w_small = 1.0 + alpha * (w_small_target - 1.0)
        w_large = 1.0 + alpha * (w_large_target - 1.0)

        # Optionally apply guards based on domain-specific val history
        w_small = guard_adjustments_small(val_hist[small_domain], w_small, small_domain)

        # Log adjustments
        lr_cur = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}/{MAX_EPOCHS} | LR={lr_cur:.6f} | w_{large_domain}={w_large:.3f} | w_{small_domain}={w_small:.3f}")

        # Iterators (oversample the smaller domain, full pass on larger domain)
        train_loaders = {'P3': train_loader_p3, 'AVO': train_loader_avo}
        itr_large = iter(train_loaders[large_domain])
        itr_small = iter(train_loaders[small_domain]) if len(train_loaders[small_domain]) > 0 else None

        steps = 0
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        while True:
            try:
                xb_large, yb_large = next(itr_large)
            except StopIteration:
                break  # finished full pass on the larger domain

            # Get a batch from smaller domain (oversampling)
            if itr_small is None:
                xb_small = None
                yb_small = None
            else:
                try:
                    xb_small, yb_small = next(itr_small)
                except StopIteration:
                    itr_small = iter(train_loaders[small_domain])
                    xb_small, yb_small = next(itr_small) if len(train_loaders[small_domain]) > 0 else (None, None)

            optimizer.zero_grad()

            # Forward on large domain
            restore_bn_buffers(model, bn_store[large_domain])
            x_large = normalize_data(xb_large).to(device)
            y_large = yb_large.to(device)
            scores_large = model(x_large)
            if scores_large.ndim > 2:
                scores_large = scores_large.view(scores_large.size(0), -1)
            loss_large = F.cross_entropy(scores_large, y_large)
            bn_store[large_domain] = snapshot_bn_buffers(model)

            # Forward on small domain
            loss_small = torch.tensor(0.0, device=device)
            if xb_small is not None:
                restore_bn_buffers(model, bn_store[small_domain])
                x_small = normalize_data(xb_small).to(device)
                y_small = yb_small.to(device)
                scores_small = model(x_small)
                if scores_small.ndim > 2:
                    scores_small = scores_small.view(scores_small.size(0), -1)
                loss_small = F.cross_entropy(scores_small, y_small)
                bn_store[small_domain] = snapshot_bn_buffers(model)

            # ABLATION 3: No MMD alignment
            total_loss = w_large * loss_large + w_small * loss_small
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning("Encountered NaN/Inf loss; skipping step and reducing LR by 2x")
                for pg in optimizer.param_groups:
                    pg['lr'] = max(1e-6, pg['lr'] * 0.5)
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Basic training stats (on large domain batch)
            with torch.no_grad():
                _, pred = scores_large.max(1)
                epoch_correct += (pred == y_large).sum().item()
                epoch_total += y_large.size(0)
                epoch_loss += total_loss.item()
            steps += 1

        # End epoch
        scheduler.step()
        avg_tr_loss = epoch_loss / max(1, steps)
        tr_acc = epoch_correct / max(1, epoch_total)
        p3_val = evaluate_domain(model, val_loader_p3, device)
        avo_val = evaluate_domain(model, val_loader_avo, device)
        val_hist['P3'].append(p3_val)
        val_hist['AVO'].append(avo_val)

        logger.info(f"Epoch Summary: TrainLoss={avg_tr_loss:.4f} | TrainAcc={tr_acc:.3f} | Val(P3)={p3_val:.3f} | Val(AVO)={avo_val:.3f}")

        # Early stopping on small-domain val
        small_val = p3_val if small_domain == 'P3' else avo_val
        improved = small_val > best_p3_val + 1e-4
        if improved:
            best_p3_val = small_val
            best_state = model.state_dict()
            patience_cnt = 0
            logger.info(f"New best {small_domain}-val; checkpoint updated")
        else:
            patience_cnt += 1
            remaining = max(0, EARLY_STOPPING_PATIENCE - patience_cnt)
            logger.info(f"No {small_domain}-val improvement; patience remaining: {remaining}/{EARLY_STOPPING_PATIENCE}")
            if patience_cnt >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered ({small_domain}-val)")
                break

    # Load best & evaluate
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def run_nested_cv_ablation3(logger: logging.Logger, channels: List[str]) -> Dict[str, float]:
    from sklearn.model_selection import StratifiedKFold
    import scipy.stats as stats

    # Load combined arrays
    X_all, y_all, src_all = load_combined_arrays(logger, channels)

    # CV config
    from config import NESTED_CV_OUTER_FOLDS, NESTED_CV_REPEATS, NESTED_CV_CONFIDENCE_LEVEL

    fold_acc = []
    dataset_metrics = {'P3': [], 'AVO': []}

    # Detailed results for statistical testing - each fold's raw results
    detailed_fold_results = []

    for repeat in range(NESTED_CV_REPEATS):
        logger.info(f"Repeat {repeat + 1}/{NESTED_CV_REPEATS}")
        cv = StratifiedKFold(n_splits=NESTED_CV_OUTER_FOLDS, shuffle=True, random_state=seeds[repeat % len(seeds)])
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_all, y_all)):
            logger.info(f"  Fold {fold_idx + 1}/{NESTED_CV_OUTER_FOLDS}")

            # Split into train/val within train fold
            X_tr_fold, y_tr_fold, src_tr_fold = X_all[train_idx], y_all[train_idx], src_all[train_idx]
            X_te_fold, y_te_fold, src_te_fold = X_all[test_idx], y_all[test_idx], src_all[test_idx]

            # Train/val split ratio same as nested_cv: TRAIN_SIZE/(TRAIN_SIZE+VAL_SIZE)
            train_val_total = TRAIN_SIZE + VAL_SIZE
            train_ratio_within = TRAIN_SIZE / train_val_total if train_val_total > 0 else 0.875
            idx_range = np.arange(len(X_tr_fold))
            tr_idx, va_idx = train_test_split(idx_range, train_size=train_ratio_within, stratify=y_tr_fold, random_state=42)

            # Build per-domain splits
            # Train per domain
            tr_mask_p3 = (src_tr_fold == 'P3')
            tr_mask_avo = (src_tr_fold == 'AVO')
            Xtr_p3 = X_tr_fold[np.intersect1d(np.where(tr_mask_p3)[0], tr_idx)]
            ytr_p3 = y_tr_fold[np.intersect1d(np.where(tr_mask_p3)[0], tr_idx)]
            Xtr_avo = X_tr_fold[np.intersect1d(np.where(tr_mask_avo)[0], tr_idx)]
            ytr_avo = y_tr_fold[np.intersect1d(np.where(tr_mask_avo)[0], tr_idx)]

            # Val per domain
            va_mask_p3 = (src_tr_fold == 'P3')
            va_mask_avo = (src_tr_fold == 'AVO')
            Xva_p3 = X_tr_fold[np.intersect1d(np.where(va_mask_p3)[0], va_idx)]
            yva_p3 = y_tr_fold[np.intersect1d(np.where(va_mask_p3)[0], va_idx)]
            Xva_avo = X_tr_fold[np.intersect1d(np.where(va_mask_avo)[0], va_idx)]
            yva_avo = y_tr_fold[np.intersect1d(np.where(va_mask_avo)[0], va_idx)]

            # Train fold model with ablation 3
            model = ablation3_train_fold(
                logger,
                Xtr_p3, ytr_p3, Xva_p3, yva_p3,
                Xtr_avo, ytr_avo, Xva_avo, yva_avo,
                channels, seed=seeds[0]
            )

            # Evaluate on test by domain
            device = get_device()

            def evaluate_subset_metrics(model_obj: nn.Module, X: np.ndarray, y: np.ndarray, mask: np.ndarray, dev: torch.device) -> Dict[str, float]:
                if not np.any(mask):
                    return {
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0,
                        'auc': 0.5,
                    }
                loader_local = DataLoader(
                    TensorDataset(torch.FloatTensor(X[mask]), torch.LongTensor(y[mask])),
                    batch_size=BATCH_SIZE, shuffle=False
                )
                details_local = evaluate(model_obj, loader_local, dev, return_details=True)
                return {
                    'accuracy': float(details_local.get('accuracy', 0.0)),
                    'precision': float(details_local.get('precision', 0.0)),
                    'recall': float(details_local.get('recall', 0.0)),
                    'f1_score': float(details_local.get('f1_score', 0.0)),
                    'auc': float(details_local.get('auc', 0.5)),
                }

            mask_p3 = (src_te_fold == 'P3')
            mask_avo = (src_te_fold == 'AVO')
            m_p3 = evaluate_subset_metrics(model, X_te_fold, y_te_fold, mask_p3, device)
            m_avo = evaluate_subset_metrics(model, X_te_fold, y_te_fold, mask_avo, device)
            n_p3 = int(np.sum(mask_p3))
            n_avo = int(np.sum(mask_avo))
            acc_overall = (m_p3['accuracy'] * n_p3 + m_avo['accuracy'] * n_avo) / max(1, (n_p3 + n_avo))

            logger.info(
                f"    P3 Test | Acc={m_p3['accuracy']:.4f} P={m_p3['precision']:.4f} R={m_p3['recall']:.4f} F1={m_p3['f1_score']:.4f} AUC={m_p3['auc']:.4f} (n={n_p3})"
            )
            logger.info(
                f"    AVO Test | Acc={m_avo['accuracy']:.4f} P={m_avo['precision']:.4f} R={m_avo['recall']:.4f} F1={m_avo['f1_score']:.4f} AUC={m_avo['auc']:.4f} (n={n_avo})"
            )
            logger.info(f"    Overall Test Acc={acc_overall:.4f}")

            fold_acc.append(acc_overall)
            dataset_metrics['P3'].append(m_p3['accuracy'])
            dataset_metrics['AVO'].append(m_avo['accuracy'])

            # Store detailed fold results for t-test analysis
            fold_result = {
                'repeat': repeat + 1,
                'fold': fold_idx + 1,
                'overall_accuracy': acc_overall,
                'p3_accuracy': m_p3['accuracy'],
                'avo_accuracy': m_avo['accuracy'],
                'p3_precision': m_p3['precision'],
                'p3_recall': m_p3['recall'],
                'p3_f1': m_p3['f1_score'],
                'p3_auc': m_p3['auc'],
                'avo_precision': m_avo['precision'],
                'avo_recall': m_avo['recall'],
                'avo_f1': m_avo['f1_score'],
                'avo_auc': m_avo['auc'],
                'p3_test_size': n_p3,
                'avo_test_size': n_avo
            }
            detailed_fold_results.append(fold_result)

    # Aggregate with CI like nested_cv
    acc_array = np.array(fold_acc, dtype=float)
    mean_acc = float(np.mean(acc_array)) if acc_array.size > 0 else 0.0
    std_acc = float(np.std(acc_array, ddof=1)) if acc_array.size > 1 else 0.0
    n_samples = max(1, acc_array.size)
    t_crit = stats.t.ppf(0.5 + (0.5 * 0.95), df=max(1, n_samples - 1))
    margin = float(t_crit * (std_acc / math.sqrt(n_samples))) if n_samples > 1 else 0.0
    ci_lower = mean_acc - margin
    ci_upper = mean_acc + margin

    # Dataset-specific with CI
    p3_arr = np.array(dataset_metrics['P3'], dtype=float)
    avo_arr = np.array(dataset_metrics['AVO'], dtype=float)
    mean_p3 = float(np.mean(p3_arr)) if p3_arr.size > 0 else 0.0
    mean_avo = float(np.mean(avo_arr)) if avo_arr.size > 0 else 0.0
    std_p3 = float(np.std(p3_arr, ddof=1)) if p3_arr.size > 1 else 0.0
    std_avo = float(np.std(avo_arr, ddof=1)) if avo_arr.size > 1 else 0.0
    n_p3 = max(1, p3_arr.size)
    n_avo = max(1, avo_arr.size)
    t_p3 = stats.t.ppf(0.5 + 0.5 * 0.95, df=max(1, n_p3 - 1))
    t_avo = stats.t.ppf(0.5 + 0.5 * 0.95, df=max(1, n_avo - 1))
    margin_p3 = float(t_p3 * (std_p3 / math.sqrt(n_p3))) if n_p3 > 1 else 0.0
    margin_avo = float(t_avo * (std_avo / math.sqrt(n_avo))) if n_avo > 1 else 0.0
    p3_ci = (mean_p3 - margin_p3, mean_p3 + margin_p3)
    avo_ci = (mean_avo - margin_avo, mean_avo + margin_avo)

    # Logging similar to main.py nested CV
    log_section_header(logger, "CROSS-VALIDATION RESULTS (ABLATION 3: NO MMD)")
    logger.info(f"Overall accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    logger.info(f"P3 Dataset - Mean Accuracy: {mean_p3:.4f} | 95% CI: [{p3_ci[0]:.4f}, {p3_ci[1]:.4f}]")
    logger.info(f"AVO Dataset - Mean Accuracy: {mean_avo:.4f} | 95% CI: [{avo_ci[0]:.4f}, {avo_ci[1]:.4f}]")

    # Calculate detailed statistics for all metrics
    def calculate_metric_stats(metric_values, metric_name):
        arr = np.array(metric_values, dtype=float)
        if arr.size == 0:
            return {f'{metric_name}_mean': 0.0, f'{metric_name}_std': 0.0,
                   f'{metric_name}_ci_lower': 0.0, f'{metric_name}_ci_upper': 0.0}

        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        n_samples = max(1, arr.size)
        t_crit = stats.t.ppf(0.5 + 0.5 * 0.95, df=max(1, n_samples - 1))
        margin = float(t_crit * (std_val / math.sqrt(n_samples))) if n_samples > 1 else 0.0

        return {
            f'{metric_name}_mean': mean_val,
            f'{metric_name}_std': std_val,
            f'{metric_name}_ci_lower': mean_val - margin,
            f'{metric_name}_ci_upper': mean_val + margin
        }

    # Extract metrics for statistics calculation
    metrics_data = {
        'overall_accuracy': [r['overall_accuracy'] for r in detailed_fold_results],
        'p3_accuracy': [r['p3_accuracy'] for r in detailed_fold_results],
        'avo_accuracy': [r['avo_accuracy'] for r in detailed_fold_results],
        'p3_precision': [r['p3_precision'] for r in detailed_fold_results],
        'p3_recall': [r['p3_recall'] for r in detailed_fold_results],
        'p3_f1': [r['p3_f1'] for r in detailed_fold_results],
        'p3_auc': [r['p3_auc'] for r in detailed_fold_results],
        'avo_precision': [r['avo_precision'] for r in detailed_fold_results],
        'avo_recall': [r['avo_recall'] for r in detailed_fold_results],
        'avo_f1': [r['avo_f1'] for r in detailed_fold_results],
        'avo_auc': [r['avo_auc'] for r in detailed_fold_results]
    }

    # Calculate statistics for all metrics
    results = {}
    for metric_name, values in metrics_data.items():
        results.update(calculate_metric_stats(values, metric_name))

    # Add detailed fold results for CSV export
    results['detailed_fold_results'] = detailed_fold_results

    # Legacy compatibility
    results.update({
        'mean_accuracy': results['overall_accuracy_mean'],
        'std_accuracy': results['overall_accuracy_std'],
        'ci_lower': results['overall_accuracy_ci_lower'],
        'ci_upper': results['overall_accuracy_ci_upper'],
        'mean_p3_accuracy': results['p3_accuracy_mean'],
        'mean_avo_accuracy': results['avo_accuracy_mean'],
        'p3_ci_lower': results['p3_accuracy_ci_lower'],
        'p3_ci_upper': results['p3_accuracy_ci_upper'],
        'avo_ci_lower': results['avo_accuracy_ci_lower'],
        'avo_ci_upper': results['avo_accuracy_ci_upper']
    })

    return results


def main():
    mne.set_log_level('ERROR')
    logging.getLogger('joblib').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')

    # Reload config to get dynamic values from batch_runner
    import importlib.util
    import os

    # Force fresh import by loading config from batch_runner override path
    # This will load the temporary config.py created by batch_runner
    config_path = os.environ.get('CONFIG_OVERRIDE_PATH', os.path.join(os.getcwd(), 'config.py'))
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

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
        logger = setup_logger('Ablation3_AVOsmall_NoMMD', create_file=True)
        log_section_header(logger, "ABLATION 3: Remove RBF-MMD Alignment")

        # Log configuration summary
        log_configuration(logger, {
            'ablation': 'No MMD Alignment (lambda_MMD = 0)',
            'electrode_list': electrode_list,
            'batch_size': BATCH_SIZE,
            'max_epochs': MAX_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'trials_per_subject_P3': config.NESTED_CV_TRIALS_PER_SUBJECT_P3,
            'trials_per_subject_AVO': config.NESTED_CV_TRIALS_PER_SUBJECT_AVO,
            'device_mode': DEVICE_MODE,
        })

        # Run Ablation 3 with proper cross-validation
        log_section_header(logger, "Running Nested Cross-Validation with Ablation 3")
        results = run_nested_cv_ablation3(logger, channels)

        # Save detailed results to CSV for t-test analysis
        import pandas as pd
        import datetime

        if 'detailed_fold_results' in results:
            df = pd.DataFrame(results['detailed_fold_results'])
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f'ablation_results_AVOsmall/ablation3_no_mmd_detailed_{timestamp}.csv'
            df.to_csv(csv_filename, index=False)
            logger.info(f"Detailed results saved to: {csv_filename}")
            print(f"Detailed results saved to: {csv_filename}")

            # Save summary statistics
            summary_stats = {k: v for k, v in results.items() if k != 'detailed_fold_results'}
            summary_df = pd.DataFrame([summary_stats])
            summary_filename = f'ablation_results_AVOsmall/ablation3_no_mmd_summary_{timestamp}.csv'
            summary_df.to_csv(summary_filename, index=False)
            logger.info(f"Summary statistics saved to: {summary_filename}")
            print(f"Summary statistics saved to: {summary_filename}")

        print("\\n--- Ablation 3 Complete (No MMD) ---")
        print(f"Final Results: Overall Accuracy = {results.get('mean_accuracy', 0.0):.4f}")

        # Force flush and exit cleanly
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    except Exception as e:
        print(f"\\n--- Ablation 3 Experiment Failed: {e} ---")
        if logger:
            cleanup_failed_log(logger)
        raise
    except KeyboardInterrupt:
        print("\\n--- Ablation 3 Experiment Interrupted by User ---")
        if logger:
            cleanup_failed_log(logger)
        raise


if __name__ == "__main__":
    main()
