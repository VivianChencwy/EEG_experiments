"""
Enhanced caching system for EEG preprocessing with support for complex feature extraction
"""

import os
import pickle
import hashlib
import numpy as np
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import threading


class EnhancedEEGCache:
    """Advanced cache system for enhanced EEG preprocessing with multi-level caching."""

    def __init__(self, cache_dir: str = "./cache", max_cache_size_mb: int = 2000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different cache types
        self.raw_cache_dir = self.cache_dir / "raw_preprocessing"
        self.enhanced_cache_dir = self.cache_dir / "enhanced_features"
        self.metadata_cache_dir = self.cache_dir / "metadata"

        for cache_subdir in [self.raw_cache_dir, self.enhanced_cache_dir, self.metadata_cache_dir]:
            cache_subdir.mkdir(parents=True, exist_ok=True)

        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.cache_index = self._load_cache_index()
        self.lock = threading.Lock()

    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index for fast lookup and management."""
        index_file = self.cache_dir / "cache_index.pkl"
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache index: {e}")
        return {}

    def _save_cache_index(self):
        """Save cache index to disk."""
        index_file = self.cache_dir / "cache_index.pkl"
        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self.cache_index, f)
        except Exception as e:
            print(f"Warning: Could not save cache index: {e}")

    def _generate_enhanced_cache_key(self, raw_file: str, channels: list,
                                   trial_start_offset: int, trial_stop_offset: int,
                                   low_freq: float, high_freq: float,
                                   resample_freq: Optional[float],
                                   # Enhanced preprocessing parameters
                                   remove_artifacts: bool = False,
                                   baseline_correct: bool = False,
                                   extract_frequency_features: bool = False,
                                   apply_notch_filter: bool = False,
                                   notch_freqs: list = None,
                                   frequency_bands: dict = None) -> str:
        """Generate cache key that includes all preprocessing parameters."""

        # Get file modification time and size for versioning
        file_path = Path(raw_file)
        if file_path.exists():
            stat = file_path.stat()
            file_info = f"{stat.st_mtime}_{stat.st_size}"
        else:
            file_info = "unknown"

        # Create comprehensive parameter dictionary
        cache_params = {
            'file': raw_file,
            'file_info': file_info,
            'channels': sorted(channels),
            'trial_start_offset': trial_start_offset,
            'trial_stop_offset': trial_stop_offset,
            'low_freq': low_freq,
            'high_freq': high_freq,
            'resample_freq': resample_freq,
            # Enhanced preprocessing flags
            'remove_artifacts': remove_artifacts,
            'baseline_correct': baseline_correct,
            'extract_frequency_features': extract_frequency_features,
            'apply_notch_filter': apply_notch_filter,
            'notch_freqs': sorted(notch_freqs) if notch_freqs else [],
            'frequency_bands': frequency_bands or {},
            'cache_version': '2.0'  # Version for cache compatibility
        }

        # Create hash
        param_str = str(sorted(cache_params.items()))
        cache_key = hashlib.sha256(param_str.encode()).hexdigest()[:16]  # Shorter hash
        return cache_key

    def _get_cache_path(self, cache_key: str, cache_type: str = 'enhanced') -> Path:
        """Get cache file path for a given key and type."""
        if cache_type == 'enhanced':
            return self.enhanced_cache_dir / f"{cache_key}.pkl"
        elif cache_type == 'raw':
            return self.raw_cache_dir / f"{cache_key}.pkl"
        else:
            return self.metadata_cache_dir / f"{cache_key}.pkl"

    def get_cached_data(self, raw_file: str, channels: list,
                       trial_start_offset: int, trial_stop_offset: int,
                       low_freq: float, high_freq: float,
                       resample_freq: Optional[float] = None,
                       **enhancement_params) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve cached enhanced preprocessing data if available."""

        cache_key = self._generate_enhanced_cache_key(
            raw_file, channels, trial_start_offset, trial_stop_offset,
            low_freq, high_freq, resample_freq, **enhancement_params
        )

        cache_path = self._get_cache_path(cache_key, 'enhanced')

        if cache_path.exists():
            try:
                start_time = time.time()

                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                windows_data = cached_data['windows_data']
                windows_labels = cached_data['windows_labels']

                load_time = time.time() - start_time

                # Update cache index
                with self.lock:
                    self.cache_index[cache_key] = {
                        'file_path': str(cache_path),
                        'file_size': cache_path.stat().st_size,
                        'last_accessed': time.time(),
                        'access_count': self.cache_index.get(cache_key, {}).get('access_count', 0) + 1,
                        'data_shape': windows_data.shape,
                        'creation_time': cached_data.get('creation_time', time.time())
                    }

                print(f"✓ Loaded cached enhanced data: {len(windows_data)} windows in {load_time:.2f}s")
                return windows_data, windows_labels

            except Exception as e:
                print(f"Error loading cache {cache_key}: {e}")
                # Remove corrupted cache file
                cache_path.unlink(missing_ok=True)
                if cache_key in self.cache_index:
                    del self.cache_index[cache_key]

        return None

    def cache_data(self, raw_file: str, channels: list,
                   trial_start_offset: int, trial_stop_offset: int,
                   low_freq: float, high_freq: float,
                   windows_data: np.ndarray, windows_labels: np.ndarray,
                   resample_freq: Optional[float] = None,
                   **enhancement_params) -> None:
        """Cache enhanced preprocessing data with compression and metadata."""

        cache_key = self._generate_enhanced_cache_key(
            raw_file, channels, trial_start_offset, trial_stop_offset,
            low_freq, high_freq, resample_freq, **enhancement_params
        )

        cache_path = self._get_cache_path(cache_key, 'enhanced')

        try:
            start_time = time.time()

            # Create comprehensive cached data
            cached_data = {
                'windows_data': windows_data,
                'windows_labels': windows_labels,
                'creation_time': time.time(),
                'preprocessing_params': enhancement_params,
                'metadata': {
                    'raw_file': raw_file,
                    'channels': channels,
                    'n_windows': len(windows_data),
                    'original_channels': len(channels),
                    'final_channels': windows_data.shape[1],
                    'window_size': windows_data.shape[2],
                    'enhancement_ratio': windows_data.shape[1] / len(channels) if len(channels) > 0 else 1
                }
            }

            # Compress and save
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            file_size = cache_path.stat().st_size
            save_time = time.time() - start_time

            # Update cache index
            with self.lock:
                self.cache_index[cache_key] = {
                    'file_path': str(cache_path),
                    'file_size': file_size,
                    'last_accessed': time.time(),
                    'access_count': 1,
                    'data_shape': windows_data.shape,
                    'creation_time': cached_data['creation_time']
                }
                self._save_cache_index()

            size_mb = file_size / (1024 * 1024)
            print(f"✓ Cached enhanced data: {len(windows_data)} windows, "
                  f"{size_mb:.1f}MB in {save_time:.2f}s -> {cache_path.name}")

            # Check if we need to clean up old cache files
            self._cleanup_if_needed()

        except Exception as e:
            print(f"Error caching data: {e}")

    def _cleanup_if_needed(self):
        """Clean up old cache files if total cache size exceeds limit."""
        total_size = sum(info.get('file_size', 0) for info in self.cache_index.values())

        if total_size > self.max_cache_size_bytes:
            print(f"Cache size {total_size/1024/1024:.1f}MB exceeds limit, cleaning up...")

            # Sort by last access time (least recently used first)
            sorted_items = sorted(
                self.cache_index.items(),
                key=lambda x: x[1].get('last_accessed', 0)
            )

            # Remove oldest entries until we're under the limit
            for cache_key, info in sorted_items:
                if total_size <= self.max_cache_size_bytes * 0.8:  # Clean to 80% of limit
                    break

                file_path = Path(info['file_path'])
                if file_path.exists():
                    file_size = info.get('file_size', 0)
                    file_path.unlink()
                    total_size -= file_size
                    print(f"Removed cache file: {file_path.name} ({file_size/1024/1024:.1f}MB)")

                del self.cache_index[cache_key]

            self._save_cache_index()

    def clear_cache(self, cache_type: str = 'all'):
        """Clear cache files of specified type."""
        with self.lock:
            if cache_type == 'all' or cache_type == 'enhanced':
                for cache_file in self.enhanced_cache_dir.glob("*.pkl"):
                    cache_file.unlink()

            if cache_type == 'all' or cache_type == 'raw':
                for cache_file in self.raw_cache_dir.glob("*.pkl"):
                    cache_file.unlink()

            if cache_type == 'all':
                self.cache_index = {}
            else:
                # Remove entries for the specified cache type
                keys_to_remove = []
                for key, info in self.cache_index.items():
                    path = Path(info['file_path'])
                    if not path.exists():
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del self.cache_index[key]

            self._save_cache_index()

        print(f"Cache cleared: {cache_type}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            if not self.cache_index:
                return {'status': 'empty'}

            total_files = len(self.cache_index)
            total_size = sum(info.get('file_size', 0) for info in self.cache_index.values())
            avg_access_count = sum(info.get('access_count', 0) for info in self.cache_index.values()) / total_files

            # Find most and least used
            most_used = max(self.cache_index.items(), key=lambda x: x[1].get('access_count', 0))
            least_used = min(self.cache_index.items(), key=lambda x: x[1].get('access_count', 0))

            return {
                'total_files': total_files,
                'total_size_mb': total_size / (1024 * 1024),
                'avg_access_count': avg_access_count,
                'max_size_mb': self.max_cache_size_bytes / (1024 * 1024),
                'usage_percent': (total_size / self.max_cache_size_bytes) * 100,
                'most_used': {
                    'key': most_used[0][:8] + '...',
                    'access_count': most_used[1].get('access_count', 0),
                    'shape': most_used[1].get('data_shape')
                },
                'least_used': {
                    'key': least_used[0][:8] + '...',
                    'access_count': least_used[1].get('access_count', 0),
                    'shape': least_used[1].get('data_shape')
                }
            }

    def preload_cache_for_subjects(self, subject_files: list, preprocessing_config: dict):
        """Preload cache for multiple subjects in background (for future enhancement)."""
        # This could be implemented for batch preprocessing
        pass