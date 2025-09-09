"""
Simple replacement for EEGBIDSDataset
"""

import os
from pathlib import Path


class EEGBIDSDataset:
    """Simple BIDS dataset wrapper."""
    
    def __init__(self, data_dir, dataset=None):
        self.data_dir = data_dir
        self.dataset = dataset
        self.base_path = Path(data_dir)
        
    def get_files(self):
        """Get all files in the dataset directory."""
        files = []
        if self.base_path.exists():
            for file_path in self.base_path.rglob('*'):
                if file_path.is_file():
                    files.append(file_path)
        return files
    
    def __str__(self):
        return f"EEGBIDSDataset(data_dir='{self.data_dir}', dataset='{self.dataset}')"