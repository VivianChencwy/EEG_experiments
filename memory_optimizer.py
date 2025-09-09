"""
Memory optimization utilities for EEG data processing.
"""

import gc
import psutil
import numpy as np
from typing import Optional, List, Tuple
import sys


class MemoryMonitor:
    """Monitor and manage memory usage during processing."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)
        
    def get_memory_percent(self) -> float:
        """Get memory usage as percentage of total system memory."""
        return self.process.memory_percent()
        
    def print_memory_status(self, label: str = ""):
        """Print current memory status."""
        current_mem = self.get_memory_usage()
        mem_percent = self.get_memory_percent()
        mem_increase = current_mem - self.initial_memory
        
        print(f"Memory {label}: {current_mem:.1f} MB ({mem_percent:.1f}%), "
              f"increase: +{mem_increase:.1f} MB")
              
    def force_garbage_collect(self):
        """Force garbage collection and return freed memory."""
        before = self.get_memory_usage()
        gc.collect()
        after = self.get_memory_usage()
        freed = before - after
        if freed > 0:
            print(f"Garbage collection freed {freed:.1f} MB")
        return freed


class MemoryEfficientProcessor:
    """Memory-efficient data processing utilities."""
    
    def __init__(self, max_memory_mb: float = 1000):
        self.max_memory_mb = max_memory_mb
        self.monitor = MemoryMonitor()
        
    def process_windows_in_chunks(self, raw_data: np.ndarray, events: np.ndarray,
                                labels: np.ndarray, window_start: int, window_size: int,
                                chunk_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process windows in chunks to reduce memory usage.
        
        Args:
            raw_data: Raw EEG data (channels, timepoints)
            events: Event information
            labels: Event labels
            window_start: Window start offset
            window_size: Window size in samples
            chunk_size: Number of windows to process at once
            
        Returns:
            Tuple of (windows_data, windows_labels)
        """
        n_events = len(events)
        n_channels = raw_data.shape[0]
        
        # Pre-allocate output arrays
        windows_data = np.empty((n_events, n_channels, window_size), dtype=np.float32)
        windows_labels = np.empty(n_events, dtype=np.int32)
        
        valid_windows = 0
        
        # Process in chunks
        for chunk_start in range(0, n_events, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_events)
            chunk_events = events[chunk_start:chunk_end]
            chunk_labels = labels[chunk_start:chunk_end]
            
            # Process this chunk
            chunk_windows = []
            chunk_valid_labels = []
            
            for i, (event_sample, _, _) in enumerate(chunk_events):
                start_sample = event_sample + window_start
                end_sample = start_sample + window_size
                
                # Check bounds
                if start_sample >= 0 and end_sample <= raw_data.shape[1]:
                    window_data = raw_data[:, start_sample:end_sample]
                    chunk_windows.append(window_data)
                    chunk_valid_labels.append(chunk_labels[i])
            
            # Store valid windows from this chunk
            if chunk_windows:
                chunk_windows = np.array(chunk_windows, dtype=np.float32)
                chunk_valid_labels = np.array(chunk_valid_labels, dtype=np.int32)
                
                n_valid_chunk = len(chunk_windows)
                windows_data[valid_windows:valid_windows + n_valid_chunk] = chunk_windows
                windows_labels[valid_windows:valid_windows + n_valid_chunk] = chunk_valid_labels
                valid_windows += n_valid_chunk
                
                # Clean up chunk data
                del chunk_windows, chunk_valid_labels
                
            # Force garbage collection if memory usage is high
            if self.monitor.get_memory_usage() > self.max_memory_mb:
                self.monitor.force_garbage_collect()
        
        # Trim arrays to actual size
        if valid_windows < n_events:
            windows_data = windows_data[:valid_windows]
            windows_labels = windows_labels[:valid_windows]
            
        return windows_data, windows_labels
        
    def optimize_array_dtype(self, data: np.ndarray, target_dtype: np.dtype = np.float32) -> np.ndarray:
        """Convert array to more memory-efficient dtype."""
        if data.dtype != target_dtype:
            # Check if conversion is safe
            if target_dtype == np.float32 and data.dtype == np.float64:
                return data.astype(np.float32)
            elif target_dtype == np.int32 and data.dtype == np.int64:
                return data.astype(np.int32)
                
        return data
        
    def estimate_memory_usage(self, n_windows: int, n_channels: int, 
                            window_size: int, dtype: np.dtype = np.float32) -> float:
        """Estimate memory usage for given data dimensions."""
        bytes_per_element = np.dtype(dtype).itemsize
        total_bytes = n_windows * n_channels * window_size * bytes_per_element
        return total_bytes / (1024 * 1024)  # Convert to MB


def reduce_memory_usage(func):
    """Decorator to monitor and optimize memory usage."""
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        monitor.print_memory_status("before")
        
        result = func(*args, **kwargs)
        
        monitor.print_memory_status("after")
        monitor.force_garbage_collect()
        monitor.print_memory_status("after GC")
        
        return result
    return wrapper


class DataTypeOptimizer:
    """Optimize data types to reduce memory usage."""
    
    @staticmethod
    def optimize_eeg_data(data: np.ndarray) -> np.ndarray:
        """Optimize EEG data array for memory efficiency."""
        # Convert float64 to float32 if possible
        if data.dtype == np.float64:
            # Check if precision loss is acceptable
            converted = data.astype(np.float32)
            if np.allclose(data, converted, rtol=1e-6):
                return converted
        return data
        
    @staticmethod
    def optimize_labels(labels: np.ndarray) -> np.ndarray:
        """Optimize label array for memory efficiency."""
        if labels.dtype == np.int64:
            # Check if int32 is sufficient
            if np.all(labels >= np.iinfo(np.int32).min) and np.all(labels <= np.iinfo(np.int32).max):
                return labels.astype(np.int32)
        return labels
        
    @staticmethod
    def get_memory_usage(array: np.ndarray) -> float:
        """Get memory usage of array in MB."""
        return array.nbytes / (1024 * 1024)