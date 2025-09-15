#!/usr/bin/env python3
"""
Cache management CLI tool for EEG preprocessing
"""

import argparse
import sys
from pathlib import Path
from enhanced_cache import EnhancedEEGCache

def show_cache_stats(cache_dir="./cache"):
    """Display detailed cache statistics."""
    cache = EnhancedEEGCache(cache_dir=cache_dir)
    stats = cache.get_cache_stats()

    print("="*60)
    print("EEG Preprocessing Cache Statistics")
    print("="*60)

    if stats.get('status') == 'empty':
        print("Cache is empty.")
        return

    print(f"Cache directory: {cache_dir}")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    print(f"Max size limit: {stats['max_size_mb']:.0f} MB")
    print(f"Usage: {stats['usage_percent']:.1f}%")
    print(f"Average access count: {stats['avg_access_count']:.1f}")

    if 'most_used' in stats:
        print(f"\nMost accessed file:")
        print(f"  Key: {stats['most_used']['key']}")
        print(f"  Access count: {stats['most_used']['access_count']}")
        print(f"  Data shape: {stats['most_used']['shape']}")

    if 'least_used' in stats:
        print(f"\nLeast accessed file:")
        print(f"  Key: {stats['least_used']['key']}")
        print(f"  Access count: {stats['least_used']['access_count']}")
        print(f"  Data shape: {stats['least_used']['shape']}")

def clear_cache(cache_dir="./cache", cache_type="all"):
    """Clear cache files."""
    cache = EnhancedEEGCache(cache_dir=cache_dir)

    print(f"Clearing {cache_type} cache files in {cache_dir}...")

    # Get stats before clearing
    stats_before = cache.get_cache_stats()
    if stats_before.get('status') != 'empty':
        size_before = stats_before['total_size_mb']
        files_before = stats_before['total_files']
    else:
        size_before = 0
        files_before = 0

    # Clear cache
    cache.clear_cache(cache_type)

    print(f"✓ Cleared {files_before} files ({size_before:.2f} MB)")

def optimize_cache(cache_dir="./cache"):
    """Optimize cache by cleaning up old entries."""
    cache = EnhancedEEGCache(cache_dir=cache_dir)

    print(f"Optimizing cache in {cache_dir}...")

    # Force cleanup by setting a temporary lower limit
    original_limit = cache.max_cache_size_bytes
    cache.max_cache_size_bytes = int(original_limit * 0.7)  # Reduce to 70%

    cache._cleanup_if_needed()

    # Restore original limit
    cache.max_cache_size_bytes = original_limit

    print("✓ Cache optimization completed")

def main():
    parser = argparse.ArgumentParser(
        description="EEG Preprocessing Cache Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s stats                    # Show cache statistics
  %(prog)s clear                    # Clear all cache files
  %(prog)s clear --type enhanced    # Clear only enhanced preprocessing cache
  %(prog)s optimize                 # Optimize cache storage
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show cache statistics')
    stats_parser.add_argument('--cache-dir', default='./cache',
                            help='Cache directory path (default: ./cache)')

    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cache files')
    clear_parser.add_argument('--cache-dir', default='./cache',
                            help='Cache directory path (default: ./cache)')
    clear_parser.add_argument('--type', choices=['all', 'enhanced', 'raw'],
                            default='all', help='Type of cache to clear (default: all)')
    clear_parser.add_argument('--confirm', action='store_true',
                            help='Skip confirmation prompt')

    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize cache storage')
    optimize_parser.add_argument('--cache-dir', default='./cache',
                               help='Cache directory path (default: ./cache)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'stats':
            show_cache_stats(args.cache_dir)

        elif args.command == 'clear':
            if not args.confirm:
                response = input(f"Are you sure you want to clear {args.type} cache in {args.cache_dir}? [y/N]: ")
                if response.lower() not in ['y', 'yes']:
                    print("Operation cancelled.")
                    return 0

            clear_cache(args.cache_dir, args.type)

        elif args.command == 'optimize':
            optimize_cache(args.cache_dir)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())