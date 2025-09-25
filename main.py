"""
Main entry point for EEG experiments (FIXED VERSION)
- Removes invalid individual subject results when using nested CV
- Only shows meaningful CROSS-VALIDATION RESULTS
"""

import os
import mne
import numpy as np
import warnings
import logging
from data_utils import EEGBIDSDataset

# Import configuration and modules
from config import (
    P3_DATA_DIR, AVO_DATA_DIR, data_dir, dataset,
    use_combined_datasets, separate_subject_classification,
    electrode_list, classifier, seeds, use_subject_layer,
    # 融合和域适应配置
    ELECTRODE_FUSION_METHOD, DOMAIN_ADAPTATION_METHOD,
    ENABLE_COMPREHENSIVE_EVALUATION, ENABLE_DOMAIN_ANALYSIS, ENABLE_SMALL_SAMPLE_ANALYSIS,
    SMALL_SAMPLE_SIZES, SMALL_SAMPLE_SUBJECTS,
    USE_NESTED_CV  # Import nested CV flag
)
from constants import COMMON_CHANNELS, P3_CHANNELS, AVO_CHANNELS
from preprocessor import OddballPreprocessor
from experiment import (
    run_experiment,
    train_combined_model, train_single_dataset_model,
    run_separate_subject_experiments,
    run_experiment_with_fusion,  # 融合实验函数
    _run_nested_cv_experiment
)
from utils import calculate_statistics, print_statistics, get_channel_list
from experiment_logger import (
    setup_logger, log_section_header, log_configuration, 
    log_individual_results, log_detailed_results, log_overall_metrics,
    cleanup_failed_log
)

# Setup logging and warnings
mne.set_log_level('ERROR')
logging.getLogger('joblib').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


def main():
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
        current_electrode_list = electrode_list
        current_separate_subject_classification = separate_subject_classification
        
        # Validate configuration
        if config.use_combined_datasets:
            # 融合实验可以使用所有电极，传统实验需要公共电极
            if ELECTRODE_FUSION_METHOD == 'none' and DOMAIN_ADAPTATION_METHOD == 'none':
                # 传统实验：强制使用公共电极
                if current_electrode_list != 'common':
                    print("Warning: Forcing electrode_list to 'common' for traditional combined datasets")
                    current_electrode_list = 'common'
            else:
                # 融合实验：保持用户配置，支持所有电极
                print(f"融合实验模式: 使用 electrode_list = '{current_electrode_list}'")

            if current_separate_subject_classification:
                print("Warning: Forcing separate_subject_classification to False for combined datasets")
                current_separate_subject_classification = False
        
        # Determine dataset name for logging
        if config.use_combined_datasets:
            dataset_name = "Combined"
        elif 'P3' in config.dataset:
            dataset_name = "P3"
        elif 'ds005863' in config.dataset:
            dataset_name = "AVO"
        else:
            dataset_name = "ConfigurableExperiments"

        # Save detailed CSV results for t-test analysis (similar to main_tfdwt.py)
        def save_csv_results(results_data, experiment_name):
            import pandas as pd
            import datetime

            if 'detailed_fold_results' in results_data:
                df = pd.DataFrame(results_data['detailed_fold_results'])
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_filename = f'{experiment_name}_detailed_results_{timestamp}.csv'
                df.to_csv(csv_filename, index=False)
                logger.info(f"Detailed results saved to: {csv_filename}")
                print(f"Detailed results saved to: {csv_filename}")

                # Save summary statistics
                summary_stats = {k: v for k, v in results_data.items() if k != 'detailed_fold_results'}
                summary_df = pd.DataFrame([summary_stats])
                summary_filename = f'{experiment_name}_summary_stats_{timestamp}.csv'
                summary_df.to_csv(summary_filename, index=False)
                logger.info(f"Summary statistics saved to: {summary_filename}")
                print(f"Summary statistics saved to: {summary_filename}")
            else:
                logger.warning("No detailed fold results found for CSV export")
                print("Warning: No detailed fold results found for CSV export")

        # Results will be saved in the log directory
        log_dir = './log_0908'
        
        # Setup logger with configuration parameters
        logger = setup_logger(dataset_name, classifier, current_separate_subject_classification, current_electrode_list)

        # Log current configuration for reproducibility
        log_configuration(logger, {
            "dataset": config.dataset,
            "use_combined_datasets": config.use_combined_datasets,
            "electrode_list": current_electrode_list,
            "classifier": classifier,
            "separate_subject_classification": current_separate_subject_classification,
            "use_subject_layer": use_subject_layer,
            "seeds": seeds,
            "use_nested_cv": USE_NESTED_CV,  # Log nested CV usage
            # 融合和域适应配置
            "electrode_fusion_method": ELECTRODE_FUSION_METHOD,
            "domain_adaptation_method": DOMAIN_ADAPTATION_METHOD,
            "enable_comprehensive_evaluation": ENABLE_COMPREHENSIVE_EVALUATION,
            "enable_domain_analysis": ENABLE_DOMAIN_ANALYSIS,
            "enable_small_sample_analysis": ENABLE_SMALL_SAMPLE_ANALYSIS,
            "small_sample_sizes": SMALL_SAMPLE_SIZES,
            "small_sample_subjects": SMALL_SAMPLE_SUBJECTS
        })
        
        # Determine which electrodes to use
        if current_electrode_list == 'common':
            channels = COMMON_CHANNELS
        elif current_electrode_list == 'all':
            # 融合实验支持使用所有电极，让融合系统自动处理电极差异
            if ELECTRODE_FUSION_METHOD != 'none' or DOMAIN_ADAPTATION_METHOD != 'none':
                channels = None  # 融合系统会自动处理电极选择
            else:
                # 传统实验仍使用公共电极
                channels = COMMON_CHANNELS
        else:
            # For individual datasets, we'll determine channels within the function
            channels = COMMON_CHANNELS  # Default fallback
        
        all_accuracies = {}

        if config.use_combined_datasets:
            # Configuration: Combined datasets + pooled training
            log_section_header(logger, "Processing Combined P3 and AVO Datasets")

            # 检查是否启用融合方法
            if ELECTRODE_FUSION_METHOD != 'none' or DOMAIN_ADAPTATION_METHOD != 'none':
                logger.info(f"启用融合方法: {ELECTRODE_FUSION_METHOD}, 域适应: {DOMAIN_ADAPTATION_METHOD}")
                results = run_experiment_with_fusion(
                    datasets=['P3', 'AVO'],
                    fusion_method=ELECTRODE_FUSION_METHOD,
                    domain_adaptation=DOMAIN_ADAPTATION_METHOD,
                    channels=channels,
                    logger=logger
                )
            else:
                logger.info("使用传统方法（无融合）")
                results = run_experiment(
                    datasets=['P3', 'AVO'],
                    training_mode='pooled',
                    channels=channels,
                    logger=logger,
                    p3_dir=P3_DATA_DIR,
                    avo_dir=AVO_DATA_DIR,
                    classifier=classifier,
                    seeds=seeds
                )
            # Handle variable return values based on experiment type
            if ELECTRODE_FUSION_METHOD != 'none' or DOMAIN_ADAPTATION_METHOD != 'none':
                # 处理融合实验结果（字典格式）
                if isinstance(results, dict):
                    combined_accuracies = results.get('accuracies', {})
                    combined_trial_counts = results.get('trial_counts', {})
                    combined_prediction_details = results.get('prediction_details', {})
                    combined_true_labels = results.get('true_labels', {})
                    combined_predictions = results.get('predictions', {})
                else:
                    # 如果返回元组格式，按传统方式处理
                    if len(results) == 6:  # Nested CV returns 6 values (including nested_results as last element)
                        combined_accuracies, combined_trial_counts, combined_prediction_details, combined_true_labels, combined_predictions, nested_results = results
                    elif len(results) == 5:
                        combined_accuracies, combined_trial_counts, combined_prediction_details, combined_true_labels, combined_predictions = results
                        nested_results = None
                    else:
                        combined_accuracies, combined_trial_counts, combined_prediction_details, combined_true_labels, combined_predictions = results
                        nested_results = None
            else:
                # 处理传统实验结果（元组格式）
                if len(results) == 6:  # Nested CV returns 6 values (including nested_results as last element)
                    combined_accuracies, combined_trial_counts, combined_prediction_details, combined_true_labels, combined_predictions, nested_results = results
                elif len(results) == 5:
                    combined_accuracies, combined_trial_counts, combined_prediction_details, combined_true_labels, combined_predictions = results
                    nested_results = None
                else:
                    combined_accuracies, combined_trial_counts, combined_prediction_details, combined_true_labels, combined_predictions = results
                    nested_results = None
            
            if combined_accuracies:
                # FIXED: Skip individual subject results if using nested CV
                if not USE_NESTED_CV:
                    # Log individual subject results only for traditional methods
                    log_section_header(logger, "Individual Subject Results - Combined Model")
                    for subject_id in sorted(combined_prediction_details.keys()):
                        log_detailed_results(logger, dataset_name, subject_id, combined_prediction_details[subject_id])
                
                    # Print aggregate statistics
                    stats_overall = calculate_statistics(combined_accuracies)
                    print_statistics(stats_overall, "Combined Model (All Subjects)", logger, combined_prediction_details)
                    
                    # Analyze P3 and AVO subset performance
                    p3_subset = {k: v for k, v in combined_accuracies.items() if k.startswith('P3_')}
                    avo_subset = {k: v for k, v in combined_accuracies.items() if k.startswith('AVO_')}
                    p3_details_subset = {k: v for k, v in combined_prediction_details.items() if k.startswith('P3_')}
                    avo_details_subset = {k: v for k, v in combined_prediction_details.items() if k.startswith('AVO_')}
                    if p3_subset:
                        print_statistics(calculate_statistics(p3_subset), "Combined Model – P3 Subjects", logger, p3_details_subset)
                    if avo_subset:
                        print_statistics(calculate_statistics(avo_subset), "Combined Model – AVO Subjects", logger, avo_details_subset)
                    all_accuracies['Combined'] = stats_overall
                else:
                    # For nested CV, the meaningful results are already logged by the nested CV framework
                    logger.info("Individual subject results skipped - using Nested Cross-Validation results")

                    # Save CSV results for t-test analysis
                    if nested_results is not None and isinstance(nested_results, dict):
                        save_csv_results(nested_results, 'main_combined')

                    # Check if dataset-specific results are available from nested CV
                    if nested_results is not None and isinstance(nested_results, dict) and 'dataset_specific_results' in nested_results:
                        dataset_results = nested_results['dataset_specific_results']
                        logger.info("Dataset-specific results from Nested Cross-Validation:")

                        # Display P3 results if available (handle both string and numpy string keys)
                        p3_key = None
                        avo_key = None
                        for key in dataset_results.keys():
                            if str(key) == 'P3':
                                p3_key = key
                            elif str(key) == 'AVO':
                                avo_key = key
                        
                        if p3_key is not None:
                            p3_stats = dataset_results[p3_key]
                            logger.info(f"P3 Dataset - Nested CV Results:")
                            logger.info(f"  Mean Accuracy: {p3_stats.get('mean_accuracy', 0):.4f} ± {p3_stats.get('std_accuracy', 0):.4f}")
                            if 'accuracy' in p3_stats:
                                logger.info(f"  95% CI: [{p3_stats['accuracy']['ci_lower']:.4f}, {p3_stats['accuracy']['ci_upper']:.4f}]")
                            if 'auc' in p3_stats:
                                logger.info(f"  AUC: {p3_stats['auc']['mean']:.4f} ± {p3_stats['auc']['std']:.4f}")

                        # Display AVO results if available
                        if avo_key is not None:
                            avo_stats = dataset_results[avo_key]
                            logger.info(f"AVO Dataset - Nested CV Results:")
                            logger.info(f"  Mean Accuracy: {avo_stats.get('mean_accuracy', 0):.4f} ± {avo_stats.get('std_accuracy', 0):.4f}")
                            if 'accuracy' in avo_stats:
                                logger.info(f"  95% CI: [{avo_stats['accuracy']['ci_lower']:.4f}, {avo_stats['accuracy']['ci_upper']:.4f}]")
                            if 'auc' in avo_stats:
                                logger.info(f"  AUC: {avo_stats['auc']['mean']:.4f} ± {avo_stats['auc']['std']:.4f}")
                    else:
                        logger.info("Dataset-specific analysis not available - check if combined datasets are being used")

                # 处理融合实验的额外结果
                if ELECTRODE_FUSION_METHOD != 'none' or DOMAIN_ADAPTATION_METHOD != 'none':
                    if isinstance(results, dict):
                        # 记录融合方法信息
                        logger.info(f"融合方法: {results.get('fusion_method', 'unknown')}")
                        logger.info(f"域适应方法: {results.get('domain_adaptation', 'unknown')}")

                        # 小样本分析结果
                        if ENABLE_SMALL_SAMPLE_ANALYSIS and 'comprehensive_analysis' in results:
                            log_section_header(logger, "小样本学习分析")
                            comprehensive_analysis = results['comprehensive_analysis']
                            if 'small_sample_analysis' in comprehensive_analysis:
                                small_sample_results = comprehensive_analysis['small_sample_analysis']
                                logger.info("小样本学习曲线分析完成")
                                # 记录关键指标
                                if 'learning_curves' in small_sample_results:
                                    for sample_config, performance in small_sample_results['learning_curves'].items():
                                        logger.info(f"样本配置 {sample_config}: 准确率 {performance:.3f}")

                        # 域间分析结果
                        if ENABLE_DOMAIN_ANALYSIS and 'comprehensive_analysis' in results:
                            comprehensive_analysis = results['comprehensive_analysis']
                            if 'domain_analysis' in comprehensive_analysis:
                                logger.info("域间分析已完成")
                                domain_analysis = comprehensive_analysis['domain_analysis']
                                if 'domain_distances' in domain_analysis:
                                    logger.info(f"域间距离: {domain_analysis['domain_distances']}")

                        # 模型参数信息
                        if 'model_params' in results:
                            logger.info(f"模型参数数量: {results['model_params']:,}")

        elif 'P3' in config.dataset:
            log_section_header(logger, "Processing P3 Dataset")
            p3_channels = P3_CHANNELS if current_electrode_list == 'all' else COMMON_CHANNELS
            
            if current_separate_subject_classification:
                # Configuration: P3 dataset + individual training
                results = run_experiment(
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
                results = run_experiment(
                    datasets=['P3'],
                    training_mode='pooled',
                    channels=p3_channels, 
                    logger=logger,
                    p3_dir=data_dir,
                    classifier=classifier,
                    seeds=seeds
                )
            
            # Handle variable return values (5 for separate, 6 for pooled)
            if len(results) == 6:
                p3_accuracies, p3_trial_counts, p3_prediction_details, p3_true_labels, p3_predictions, _ = results
            else:
                p3_accuracies, p3_trial_counts, p3_prediction_details, p3_true_labels, p3_predictions = results
            
            if p3_accuracies:
                # FIXED: Skip individual subject results if using nested CV
                if not USE_NESTED_CV:
                    # Log individual subject results only for traditional methods
                    log_section_header(logger, "Individual Subject Results - P3")
                    for subject_id in sorted(p3_prediction_details.keys()):
                        log_detailed_results(logger, dataset_name, subject_id, p3_prediction_details[subject_id])
                    
                    # Print aggregate statistics
                    stats = calculate_statistics(p3_accuracies)
                    model_type = "Individual Models" if current_separate_subject_classification else "Pooled Model"
                    print_statistics(stats, f"P3 {model_type}", logger, p3_prediction_details)
                    all_accuracies['P3'] = stats
                else:
                    # For nested CV, the meaningful results are already logged by the nested CV framework
                    logger.info("Individual subject results skipped - using Nested Cross-Validation results")

                    # Save CSV results for t-test analysis
                    if len(results) == 6 and results[5] is not None:  # nested_results is available
                        nested_results_p3 = results[5]
                        if isinstance(nested_results_p3, dict):
                            save_csv_results(nested_results_p3, 'main_P3')

        elif 'ds005863' in config.dataset:
            log_section_header(logger, "Processing Active Visual Oddball Dataset")
            avo_channels = AVO_CHANNELS if current_electrode_list == 'all' else COMMON_CHANNELS
            
            if current_separate_subject_classification:
                # Configuration: AVO dataset + individual training
                results = run_experiment(
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
                results = run_experiment(
                    datasets=['AVO'],
                    training_mode='pooled',
                    channels=avo_channels,
                    logger=logger,
                    avo_dir=data_dir,
                    classifier=classifier,
                    seeds=seeds
                )
            
            # Handle variable return values (5 for separate, 6 for pooled)
            if len(results) == 6:
                avo_accuracies, avo_trial_counts, avo_prediction_details, avo_true_labels, avo_predictions, _ = results
            else:
                avo_accuracies, avo_trial_counts, avo_prediction_details, avo_true_labels, avo_predictions = results
            
            if avo_accuracies:
                # FIXED: Skip individual subject results if using nested CV
                if not USE_NESTED_CV:
                    # Log individual subject results only for traditional methods
                    log_section_header(logger, "Individual Subject Results - AVO")
                    for subject_id in sorted(avo_prediction_details.keys()):
                        log_detailed_results(logger, dataset_name, subject_id, avo_prediction_details[subject_id])
                    
                    # Print aggregate statistics
                    stats = calculate_statistics(avo_accuracies)
                    model_type = "Individual Models" if current_separate_subject_classification else "Pooled Model"
                    print_statistics(stats, f"AVO {model_type}", logger, avo_prediction_details)
                    all_accuracies['AVO'] = stats
                else:
                    # For nested CV, the meaningful results are already logged by the nested CV framework
                    logger.info("Individual subject results skipped - using Nested Cross-Validation results")

                    # Save CSV results for t-test analysis
                    if len(results) == 6 and results[5] is not None:  # nested_results is available
                        nested_results_avo = results[5]
                        if isinstance(nested_results_avo, dict):
                            save_csv_results(nested_results_avo, 'main_AVO')

        print("\n--- Experiment Run Complete ---")
        
    except Exception as e:
        print(f"\n--- Experiment Failed: {e} ---")
        if logger:
            cleanup_failed_log(logger)
        raise  # Re-raise the exception to maintain proper exit code
    except KeyboardInterrupt:
        print("\n--- Experiment Interrupted by User ---")
        if logger:
            cleanup_failed_log(logger)
        raise


if __name__ == "__main__":
    main()