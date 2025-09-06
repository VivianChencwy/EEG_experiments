"""
Experiment logger for tracking and logging experiment results.
"""
import logging
from datetime import datetime
import numpy as np
import os
from config import LOG_DIR

def setup_logger(experiment_type, classifier=None, separate_subject_classification=None, electrode_list=None, create_file=True):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create log directory if it doesn't exist
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    # Create descriptive filename with configuration parameters
    if classifier and separate_subject_classification is not None and electrode_list:
        logfile = os.path.join(log_dir, f'{experiment_type}_clf-{classifier}_sep-{separate_subject_classification}_el-{electrode_list}_results_{timestamp}.log')
    else:
        logfile = os.path.join(log_dir, f'{experiment_type}_results_{timestamp}.log')

    # Only create file handler if requested
    handlers = [logging.StreamHandler()]
    if create_file:
        handlers.append(logging.FileHandler(logfile))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=handlers,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Python>=3.8
    )

    # Return a named logger (avoids duplicate handlers if caller also uses logging).
    logger = logging.getLogger(experiment_type)
    
    # Store the log file path for potential cleanup
    if create_file:
        logger.log_file_path = logfile
    
    return logger

def cleanup_failed_log(logger):
    """Clean up log file if experiment failed."""
    if hasattr(logger, 'log_file_path') and os.path.exists(logger.log_file_path):
        try:
            os.remove(logger.log_file_path)
            print(f"Cleaned up failed experiment log: {logger.log_file_path}")
        except Exception as e:
            print(f"Failed to clean up log file {logger.log_file_path}: {e}")

def log_section_header(logger, title):
    logger.info("\n" + "="*50)
    logger.info(title)
    logger.info("="*50)

def log_individual_results(logger, experiment_type, subject_id, accuracy):
    logger.info(f"Subject: {subject_id}, Accuracy: {accuracy:.3%}")


def log_detailed_results(logger, experiment_type, subject_id, metrics):
    """Log detailed metrics including accuracy, precision, recall, f1 score, AUC and confusion matrix stats."""
    logger.info(f"Subject: {subject_id}")
    logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.3%}")
    logger.info(f"  Precision: {metrics.get('precision', 0):.3f}")
    logger.info(f"  Recall: {metrics.get('recall', 0):.3f}")
    logger.info(f"  F1 Score: {metrics.get('f1_score', 0):.3f}")
    logger.info(f"  AUC: {metrics.get('auc', 0):.3f}")
    logger.info(f"  Correct/Total: {metrics.get('correct_count', 0)}/{metrics.get('total_count', 0)}")
    logger.info(f"  Confusion Matrix Stats:")
    logger.info(f"    TP: {metrics.get('tp', 0)}, TN: {metrics.get('tn', 0)}")
    logger.info(f"    FP: {metrics.get('fp', 0)}, FN: {metrics.get('fn', 0)}")

def log_error(logger, experiment_type, subject_id, error_msg):
    logger.error(f"\nError in {experiment_type} - Subject {subject_id}:")
    logger.error(str(error_msg))

def log_configuration(logger, config_dict):
    logger.info("\nExperiment Configuration:")
    logger.info("-" * 50)
    for key, value in config_dict.items():
        logger.info(f"{key}: {value}")
    logger.info("-" * 50)

def log_overall_metrics(logger, metrics, confusion_matrix_path=None):
    """Log overall experiment metrics and confusion matrix location."""
    logger.info("\nOverall Experiment Metrics:")
    logger.info("-" * 50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    if confusion_matrix_path:
        logger.info(f"\nConfusion Matrix Plot: {confusion_matrix_path}")
    logger.info("-" * 50)
