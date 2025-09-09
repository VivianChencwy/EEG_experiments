"""
Parallel processing utilities for EEG data processing.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import numpy as np
from typing import List, Tuple, Dict, Any
import time


def process_single_subject_data(args):
    """
    Process a single subject's data - designed to be called by parallel workers.
    
    Args:
        args: tuple containing (subject_data, preprocessor, channels, subject_id)
    
    Returns:
        tuple: (subject_id, processed_data, labels, processing_time)
    """
    subject_data, preprocessor, channels, subject_id = args
    
    start_time = time.time()
    
    try:
        # Process the subject's raw data
        dataset = preprocessor.transform(subject_data)
        
        # Extract data and labels
        processed_data = dataset.data
        labels = dataset.labels
        
        processing_time = time.time() - start_time
        
        return subject_id, processed_data, labels, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"Error processing subject {subject_id}: {e}")
        return subject_id, None, None, processing_time


def process_subjects_parallel(subject_data_list: List[Tuple], 
                            preprocessor,
                            channels: List[str],
                            n_workers: int = None,
                            use_threads: bool = False) -> Dict[str, Any]:
    """
    Process multiple subjects' data in parallel.
    
    Args:
        subject_data_list: List of tuples (subject_id, raw_data)
        preprocessor: Preprocessor instance
        channels: List of channel names
        n_workers: Number of parallel workers (default: CPU count)
        use_threads: Use ThreadPoolExecutor instead of ProcessPoolExecutor
    
    Returns:
        dict: Results dictionary with processed data for each subject
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
        
    print(f"Processing {len(subject_data_list)} subjects using {n_workers} workers")
    
    # Prepare arguments for parallel processing
    process_args = [(raw_data, preprocessor, channels, subject_id) 
                   for subject_id, raw_data in subject_data_list]
    
    results = {}
    total_processing_time = 0
    
    start_time = time.time()
    
    # Choose executor based on use_threads parameter
    ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    try:
        with ExecutorClass(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_single_subject_data, args) 
                      for args in process_args]
            
            # Collect results as they complete
            for i, future in enumerate(futures):
                try:
                    subject_id, processed_data, labels, proc_time = future.result()
                    total_processing_time += proc_time
                    
                    if processed_data is not None:
                        results[subject_id] = {
                            'data': processed_data,
                            'labels': labels,
                            'processing_time': proc_time
                        }
                        print(f"Completed subject {subject_id} ({i+1}/{len(futures)}) - {proc_time:.2f}s")
                    else:
                        print(f"Failed to process subject {subject_id}")
                        
                except Exception as e:
                    print(f"Error collecting result for subject: {e}")
                    
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        # Fallback to sequential processing
        print("Falling back to sequential processing...")
        return process_subjects_sequential(subject_data_list, preprocessor, channels)
    
    total_time = time.time() - start_time
    
    print(f"Parallel processing completed in {total_time:.2f}s")
    print(f"Total processing time (sum): {total_processing_time:.2f}s")
    print(f"Speedup factor: {total_processing_time/total_time:.2f}x")
    
    return results


def process_subjects_sequential(subject_data_list: List[Tuple],
                              preprocessor,
                              channels: List[str]) -> Dict[str, Any]:
    """
    Process subjects sequentially (fallback method).
    
    Args:
        subject_data_list: List of tuples (subject_id, raw_data)
        preprocessor: Preprocessor instance
        channels: List of channel names
    
    Returns:
        dict: Results dictionary with processed data for each subject
    """
    print(f"Processing {len(subject_data_list)} subjects sequentially")
    
    results = {}
    start_time = time.time()
    
    for i, (subject_id, raw_data) in enumerate(subject_data_list):
        try:
            subject_start_time = time.time()
            
            # Process the subject's data
            dataset = preprocessor.transform(raw_data)
            
            # Extract data and labels
            processed_data = dataset.data
            labels = dataset.labels
            
            processing_time = time.time() - subject_start_time
            
            results[subject_id] = {
                'data': processed_data,
                'labels': labels,
                'processing_time': processing_time
            }
            
            print(f"Completed subject {subject_id} ({i+1}/{len(subject_data_list)}) - {processing_time:.2f}s")
            
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
    
    total_time = time.time() - start_time
    print(f"Sequential processing completed in {total_time:.2f}s")
    
    return results


class ParallelDataProcessor:
    """
    A class to manage parallel processing of EEG data.
    """
    
    def __init__(self, n_workers=None, use_threads=False, enable_parallel=True):
        self.n_workers = n_workers or mp.cpu_count()
        self.use_threads = use_threads
        self.enable_parallel = enable_parallel
        
    def process_subjects(self, subject_data_list, preprocessor, channels):
        """Process multiple subjects with optional parallel execution."""
        
        if self.enable_parallel and len(subject_data_list) > 1:
            return process_subjects_parallel(
                subject_data_list, 
                preprocessor, 
                channels,
                self.n_workers,
                self.use_threads
            )
        else:
            return process_subjects_sequential(
                subject_data_list,
                preprocessor,
                channels
            )
            
    def get_optimal_workers(self, n_subjects):
        """Get optimal number of workers based on number of subjects."""
        return min(self.n_workers, n_subjects, mp.cpu_count())