"""
Experiment logger for tracking and logging experiment results.
"""
import logging
from datetime import datetime
import numpy as np
import os

def setup_logger(experiment_type, classifier=None, separate_subject_classification=None, electrode_list=None):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create log directory if it doesn't exist
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create descriptive filename with configuration parameters
    if classifier and separate_subject_classification is not None and electrode_list:
        logfile = os.path.join(log_dir, f'{experiment_type}_clf-{classifier}_sep-{separate_subject_classification}_el-{electrode_list}_results_{timestamp}.log')
    else:
        logfile = os.path.join(log_dir, f'{experiment_type}_results_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ],
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Python>=3.8
    )

    # Return a named logger (avoids duplicate handlers if caller also uses logging).
    return logging.getLogger(experiment_type)

def log_section_header(logger, title):
    logger.info("\n" + "="*50)
    logger.info(title)
    logger.info("="*50)

def log_individual_results(logger, experiment_type, subject_id, accuracy):
    logger.info(f"Subject: {subject_id}, Accuracy: {accuracy:.3%}")

def log_error(logger, experiment_type, subject_id, error_msg):
    logger.error(f"\nError in {experiment_type} - Subject {subject_id}:")
    logger.error(str(error_msg))

def log_configuration(logger, config_dict):
    logger.info("\nExperiment Configuration:")
    logger.info("-" * 50)
    for key, value in config_dict.items():
        logger.info(f"{key}: {value}")
    logger.info("-" * 50)
