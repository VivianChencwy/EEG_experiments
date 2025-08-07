"""
Main entry point for EEG experiments
"""

import mne
import warnings
import logging
from eegdash.data_utils import EEGBIDSDataset

# Import configuration and modules
from config import (
    P3_DATA_DIR, AVO_DATA_DIR, data_dir, dataset, 
    use_combined_datasets, separate_subject_classification, 
    electrode_list, classifier, seeds, use_subject_layer
)
from constants import COMMON_CHANNELS, P3_CHANNELS, AVO_CHANNELS
from preprocessor import OddballPreprocessor
from experiment import (
    run_experiment,
    train_combined_model, train_single_dataset_model, 
    run_separate_subject_experiments
)
from utils import calculate_statistics, print_statistics, get_channel_list
from experiment_logger import setup_logger, log_section_header, log_configuration, log_individual_results

# Setup logging and warnings
mne.set_log_level('ERROR')
logging.getLogger('joblib').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


def main():
    current_electrode_list = electrode_list
    current_separate_subject_classification = separate_subject_classification
    
    # Validate configuration
    if use_combined_datasets:
        if current_electrode_list != 'common':
            print("Warning: Forcing electrode_list to 'common' for combined datasets")
            current_electrode_list = 'common'
        if current_separate_subject_classification:
            print("Warning: Forcing separate_subject_classification to False for combined datasets")
            current_separate_subject_classification = False
    
    # Determine dataset name for logging
    if use_combined_datasets:
        dataset_name = "Combined"
    elif 'P3' in dataset:
        dataset_name = "P3"
    elif 'ds005863' in dataset:
        dataset_name = "AVO"
    else:
        dataset_name = "ConfigurableExperiments"
    
    # Setup logger with configuration parameters
    logger = setup_logger(dataset_name, classifier, current_separate_subject_classification, current_electrode_list)

    # Log current configuration for reproducibility
    log_configuration(logger, {
        "dataset": dataset,
        "use_combined_datasets": use_combined_datasets,
        "electrode_list": current_electrode_list,
        "classifier": classifier,
        "separate_subject_classification": current_separate_subject_classification,
        "use_subject_layer": use_subject_layer,
        "seeds": seeds
    })
    
    # Determine which electrodes to use
    if current_electrode_list == 'common':
        channels = COMMON_CHANNELS
    else:
        # For individual datasets, we'll determine channels within the function
        channels = COMMON_CHANNELS  # Default for combined datasets
    
    all_accuracies = {}

    if use_combined_datasets:
        # Configuration: Combined datasets + pooled training
        log_section_header(logger, "Processing Combined P3 and AVO Datasets")
        combined_accuracies = run_experiment(
            datasets=['P3', 'AVO'],
            training_mode='pooled', 
            channels=channels,
            logger=logger,
            p3_dir=P3_DATA_DIR,
            avo_dir=AVO_DATA_DIR,
            classifier=classifier,
            seeds=seeds
        )
        
        if combined_accuracies:
            for subj_id, acc in combined_accuracies.items():
                log_individual_results(logger, "Combined", subj_id, acc)
            stats_overall = calculate_statistics(combined_accuracies)
            print_statistics(stats_overall, "Combined Model (All Subjects)", logger)
            
            # Analyze P3 and AVO subset performance
            p3_subset = {k: v for k, v in combined_accuracies.items() if k.startswith('P3_')}
            avo_subset = {k: v for k, v in combined_accuracies.items() if k.startswith('AVO_')}
            if p3_subset:
                print_statistics(calculate_statistics(p3_subset), "Combined Model – P3 Subjects", logger)
            if avo_subset:
                print_statistics(calculate_statistics(avo_subset), "Combined Model – AVO Subjects", logger)
            all_accuracies['Combined'] = stats_overall

    elif 'P3' in dataset:
        log_section_header(logger, "Processing P3 Dataset")
        p3_channels = P3_CHANNELS if current_electrode_list == 'all' else COMMON_CHANNELS
        
        if current_separate_subject_classification:
            # Configuration: P3 dataset + individual training
            p3_accuracies = run_experiment(
                datasets=['P3'],
                training_mode='separate',
                channels=p3_channels,
                logger=logger,
                p3_dir=data_dir,
                classifier=classifier,
                seeds=seeds
            )
        else:
            # Configuration: P3 dataset + pooled training
            p3_accuracies = run_experiment(
                datasets=['P3'],
                training_mode='pooled',
                channels=p3_channels, 
                logger=logger,
                p3_dir=data_dir,
                classifier=classifier,
                seeds=seeds
            )
        
        if p3_accuracies:
            for subj_id, acc in p3_accuracies.items():
                mode = "Individual" if current_separate_subject_classification else "Pooled"
                log_individual_results(logger, f"P3-{mode}", subj_id, acc)
            stats = calculate_statistics(p3_accuracies)
            model_type = "Individual Models" if current_separate_subject_classification else "Pooled Model"
            print_statistics(stats, f"P3 {model_type}", logger)
            all_accuracies['P3'] = stats

    elif 'ds005863' in dataset:
        log_section_header(logger, "Processing Active Visual Oddball Dataset")
        avo_channels = AVO_CHANNELS if current_electrode_list == 'all' else COMMON_CHANNELS
        
        if current_separate_subject_classification:
            # Configuration: AVO dataset + individual training
            avo_accuracies = run_experiment(
                datasets=['AVO'],
                training_mode='separate',
                channels=avo_channels,
                logger=logger,
                avo_dir=data_dir,
                classifier=classifier,
                seeds=seeds
            )
        else:
            # Configuration: AVO dataset + pooled training
            avo_accuracies = run_experiment(
                datasets=['AVO'],
                training_mode='pooled',
                channels=avo_channels,
                logger=logger,
                avo_dir=data_dir,
                classifier=classifier,
                seeds=seeds
            )
        
        if avo_accuracies:
            for subj_id, acc in avo_accuracies.items():
                mode = "Individual" if current_separate_subject_classification else "Pooled"
                log_individual_results(logger, f"AVO-{mode}", subj_id, acc)
            stats = calculate_statistics(avo_accuracies)
            model_type = "Individual Models" if current_separate_subject_classification else "Pooled Model"
            print_statistics(stats, f"AVO {model_type}", logger)
            all_accuracies['AVO'] = stats

    print("\n--- Experiment Run Complete ---")


if __name__ == "__main__":
    main() 