#!/usr/bin/env python3
"""
Cache management utility for EEG experiments.
Usage: python cache_manager.py [command]

Commands:
  info     - Show cache information
  clear    - Clear all cached data
  size     - Show cache size
"""

import sys
import argparse
from data_cache import EEGDataCache


def main():
    parser = argparse.ArgumentParser(description='EEG Data Cache Manager')
    parser.add_argument('command', choices=['info', 'clear', 'size'], 
                       help='Command to execute')
    parser.add_argument('--cache-dir', default='./cache',
                       help='Cache directory path')
    
    args = parser.parse_args()
    
    cache = EEGDataCache(cache_dir=args.cache_dir)
    
    if args.command == 'info':
        info = cache.get_cache_info()
        print(f"Cache Directory: {info['cache_dir']}")
        print(f"Number of cached files: {info['num_files']}")
        print(f"Total cache size: {info['total_size_mb']:.2f} MB")
        
    elif args.command == 'clear':
        cache.clear_cache()
        
    elif args.command == 'size':
        info = cache.get_cache_info()
        print(f"{info['total_size_mb']:.2f} MB")


if __name__ == '__main__':
    main()