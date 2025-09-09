"""
Data caching system for EEG experiments to speed up repeated data processing.
"""

import os
import pickle
import hashlib
import numpy as np
from pathlib import Path
import mne
from typing import Optional, Tuple, Any


class EEGDataCache:
    """Cache system for preprocessed EEG data."""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_cache_key(self, raw_file: str, channels: list, 
                          trial_start_offset: int, trial_stop_offset: int,
                          low_freq: float, high_freq: float, 
                          resample_freq: Optional[float] = None) -> str:
        """Generate a unique cache key for the given parameters."""
        # Get file modification time and size
        file_path = Path(raw_file)
        if file_path.exists():
            stat = file_path.stat()
            file_info = f"{stat.st_mtime}_{stat.st_size}"
        else:
            file_info = "unknown"
            
        # Create parameter string
        params = {
            'file': raw_file,
            'file_info': file_info,
            'channels': sorted(channels),
            'trial_start_offset': trial_start_offset,
            'trial_stop_offset': trial_stop_offset,
            'low_freq': low_freq,
            'high_freq': high_freq,
            'resample_freq': resample_freq
        }
        
        # Create hash
        param_str = str(params)
        cache_key = hashlib.md5(param_str.encode()).hexdigest()
        return cache_key
        
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.pkl"
        
    def get_cached_data(self, raw_file: str, channels: list,
                       trial_start_offset: int, trial_stop_offset: int,
                       low_freq: float, high_freq: float,
                       resample_freq: Optional[float] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve cached preprocessed data if available."""
        cache_key = self._generate_cache_key(
            raw_file, channels, trial_start_offset, trial_stop_offset,
            low_freq, high_freq, resample_freq
        )
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    windows_data = cached_data['windows_data']
                    windows_labels = cached_data['windows_labels']
                    print(f"Loaded cached data: {len(windows_data)} windows")
                    return windows_data, windows_labels
            except Exception as e:
                print(f"Error loading cache: {e}")
                # Remove corrupted cache file
                cache_path.unlink(missing_ok=True)
        
        return None
        
    def cache_data(self, raw_file: str, channels: list,
                   trial_start_offset: int, trial_stop_offset: int,
                   low_freq: float, high_freq: float,
                   windows_data: np.ndarray, windows_labels: np.ndarray,
                   resample_freq: Optional[float] = None) -> None:
        """Cache preprocessed data."""
        cache_key = self._generate_cache_key(
            raw_file, channels, trial_start_offset, trial_stop_offset,
            low_freq, high_freq, resample_freq
        )
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cached_data = {
                'windows_data': windows_data,
                'windows_labels': windows_labels,
                'metadata': {
                    'raw_file': raw_file,
                    'channels': channels,
                    'n_windows': len(windows_data),
                    'n_channels': windows_data.shape[1],
                    'window_size': windows_data.shape[2]
                }
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Cached data: {len(windows_data)} windows -> {cache_path.name}")
            
        except Exception as e:
            print(f"Error caching data: {e}")
            
    def clear_cache(self) -> None:
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print("Cache cleared")
        
    def get_cache_info(self) -> dict:
        """Get information about cached files."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_dir': str(self.cache_dir),
            'num_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024)
        }