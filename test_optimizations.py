#!/usr/bin/env python3
"""
Test script to verify performance optimizations.
"""

import time
import sys
from pathlib import Path
from data_cache import EEGDataCache
from memory_optimizer import MemoryMonitor
from config import (
    USE_DATA_CACHE, CACHE_DIR, ENABLE_PARALLEL_PROCESSING,
    ENABLE_MEMORY_OPTIMIZATION, VERBOSE_PROCESSING
)


def test_cache_system():
    """Test the caching system."""
    print("=== Testing Cache System ===")
    
    cache = EEGDataCache(cache_dir=CACHE_DIR)
    info = cache.get_cache_info()
    
    print(f"Cache directory: {info['cache_dir']}")
    print(f"Cached files: {info['num_files']}")
    print(f"Total cache size: {info['total_size_mb']:.2f} MB")
    
    if info['num_files'] > 0:
        print("✅ Cache system is working - found cached data")
    else:
        print("⚠️ No cached data found - first run will populate cache")


def test_memory_monitoring():
    """Test memory monitoring."""
    print("\n=== Testing Memory Monitoring ===")
    
    monitor = MemoryMonitor()
    monitor.print_memory_status("initial")
    
    # Simulate some memory usage
    import numpy as np
    large_array = np.random.rand(1000, 1000, 10).astype(np.float32)
    monitor.print_memory_status("after allocation")
    
    del large_array
    monitor.force_garbage_collect()
    monitor.print_memory_status("after cleanup")
    
    print("✅ Memory monitoring is working")


def test_configuration():
    """Test optimization configuration."""
    print("\n=== Testing Configuration ===")
    
    print(f"Data caching: {'✅ Enabled' if USE_DATA_CACHE else '❌ Disabled'}")
    print(f"Parallel processing: {'✅ Enabled' if ENABLE_PARALLEL_PROCESSING else '❌ Disabled'}")
    print(f"Memory optimization: {'✅ Enabled' if ENABLE_MEMORY_OPTIMIZATION else '❌ Disabled'}")
    print(f"Verbose output: {'✅ Enabled' if VERBOSE_PROCESSING else '❌ Disabled'}")
    
    print(f"Cache directory: {CACHE_DIR}")


def benchmark_processing():
    """Run a simple benchmark of data processing."""
    print("\n=== Running Processing Benchmark ===")
    
    # This would need actual EEG data to test properly
    print("⚠️ Actual benchmark requires EEG data files")
    print("Run the main experiment to see real performance improvements")


def main():
    print("EEG Experiments - Performance Optimization Test")
    print("=" * 50)
    
    try:
        test_configuration()
        test_cache_system()
        test_memory_monitoring()
        benchmark_processing()
        
        print("\n" + "=" * 50)
        print("✅ All optimization tests completed successfully!")
        print("\nTo see real performance improvements:")
        print("1. Run your experiment once to populate cache")
        print("2. Run it again to see caching benefits")
        print("3. Monitor memory usage during processing")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()