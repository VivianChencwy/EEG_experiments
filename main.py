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
    electrode_list, classifier, seeds
)
from constants import COMMON_CHANNELS, P3_CHANNELS, AVO_CHANNELS
from preprocessor import OddballPreprocessor
from experiment import (
    train_combined_model, train_single_dataset_model, 
    run_separate_subject_experiments
)
from utils import get_channel_list, calculate_statistics, print_statistics
from experiment_logger import setup_logger, log_section_header, log_configuration

# Setup logging and warnings
mne.set_log_level('ERROR')
logging.getLogger('joblib').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


def main():
    """Main function to run EEG experiments."""
    # Validate configuration
    if use_combined_datasets:
        if electrode_list != 'common':
            print("Warning: Forcing electrode_list to 'common' for combined datasets")
            electrode_list = 'common'
        if separate_subject_classification:
            print("Warning: Forcing separate_subject_classification to False for combined datasets")
            separate_subject_classification = False
    
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
    logger = setup_logger(dataset_name, classifier, separate_subject_classification, electrode_list)

    # Log current configuration for reproducibility
    log_configuration(logger, {
        "dataset": dataset,
        "use_combined_datasets": use_combined_datasets,
        "electrode_list": electrode_list,
        "classifier": classifier,
        "separate_subject_classification": separate_subject_classification,
        "seeds": seeds
    })
    
    all_accuracies = {}
    
    # Determine which electrodes to use
    if electrode_list == 'common':
        p3_channels = COMMON_CHANNELS
        avo_channels = COMMON_CHANNELS
    else:
        p3_channels = P3_CHANNELS
        avo_channels = AVO_CHANNELS

    # --------------------------------------------------------------------------
    # Experiment Logic
    # --------------------------------------------------------------------------

    if use_combined_datasets:
        log_section_header(logger, "Processing Combined P3 and AVO Datasets")
        avo_dataset_obj = EEGBIDSDataset(data_dir=AVO_DATA_DIR, dataset='ds005863')
        combined_accuracies = train_combined_model(P3_DATA_DIR, avo_dataset_obj, p3_channels, logger)
        
        if combined_accuracies:
            # Log individual results
            from experiment_logger import log_individual_results
            for subj_id, acc in combined_accuracies.items():
                log_individual_results(logger, "Combined", subj_id, acc)
            
            # Overall statistics
            stats_overall = calculate_statistics(combined_accuracies)
            print_statistics(stats_overall, "Combined Model (All Subjects)", logger)
            
            # Subset statistics
            p3_subset = {k: v for k, v in combined_accuracies.items() if k.startswith('P3_')}
            avo_subset = {k: v for k, v in combined_accuracies.items() if k.startswith('AVO_')}
            
            if p3_subset:
                print_statistics(calculate_statistics(p3_subset), "Combined Model – P3 Subjects", logger)
            if avo_subset:
                print_statistics(calculate_statistics(avo_subset), "Combined Model – AVO Subjects", logger)
            
            all_accuracies['Combined'] = stats_overall

    elif 'P3' in dataset:
        log_section_header(logger, "Processing P3 Dataset")
        p3_preprocessor = OddballPreprocessor(p3_channels)
        
        if separate_subject_classification:
            p3_accuracies = run_separate_subject_experiments(data_dir, p3_channels, logger, 'P3')
            
            if p3_accuracies:
                stats = calculate_statistics(p3_accuracies)
                print_statistics(stats, "P3", logger)
                all_accuracies['P3'] = stats
        else:
            # Pooled P3
            p3_pooled_accuracies = train_single_dataset_model(
                data_dir, p3_preprocessor, p3_channels, logger, 'P3'
            )
            
            if p3_pooled_accuracies:
                from experiment_logger import log_individual_results
                for subj_id, acc in p3_pooled_accuracies.items():
                    log_individual_results(logger, "P3-Pooled", subj_id, acc)
                
                stats_overall = calculate_statistics(p3_pooled_accuracies)
                print_statistics(stats_overall, "P3 Pooled Model (All Subjects)", logger)
                all_accuracies['P3-Pooled'] = stats_overall

    elif 'ds005863' in dataset:
        log_section_header(logger, "Processing Active Visual Oddball Dataset")
        avo_preprocessor = OddballPreprocessor(avo_channels)
        
        if separate_subject_classification:
            avo_accuracies = run_separate_subject_experiments(data_dir, avo_channels, logger, 'AVO')
            
            if avo_accuracies:
                stats = calculate_statistics(avo_accuracies)
                print_statistics(stats, "Active Visual Oddball", logger)
                all_accuracies['AVO'] = stats
        else:
            # Pooled AVO
            avo_pooled_accuracies = train_single_dataset_model(
                data_dir, avo_preprocessor, avo_channels, logger, 'AVO'
            )
            
            if avo_pooled_accuracies:
                from experiment_logger import log_individual_results
                for subj_id, acc in avo_pooled_accuracies.items():
                    log_individual_results(logger, "AVO-Pooled", subj_id, acc)
                
                stats_overall = calculate_statistics(avo_pooled_accuracies)
                print_statistics(stats_overall, "AVO Pooled Model (All Subjects)", logger)
                all_accuracies['AVO-Pooled'] = stats_overall
    
    # Final summary
    print("\n--- Experiment Run Complete ---")
    if all_accuracies:
        print("Summary of all results:")
        for exp_name, stats in all_accuracies.items():
            print(f"{exp_name}: {stats['mean']:.3f} ± {(stats['ci_upper'] - stats['ci_lower'])/2:.3f}")


if __name__ == "__main__":
    main() 