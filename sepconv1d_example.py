#!/usr/bin/env python3
"""
SepConv1D Model Usage Example
============================

This script demonstrates how to use the newly integrated SepConv1D model 
for EEG P300 classification in the EEG_experiments framework.

SepConv1D is specifically designed for small datasets and helps prevent
overfitting through its lightweight separable convolution architecture.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from models import create_model, train_model, SepConv1D
import torch

def demo_sepconv1d_basic():
    """
    Demonstrate basic usage of SepConv1D model
    """
    print("="*60)
    print("SepConv1D Basic Usage Demo")
    print("="*60)
    
    # Model parameters
    n_channels = 26  # For AVO dataset with selected channels
    n_times = INPUT_WINDOW_SAMPLES  # 128 samples at 128Hz (1 second)
    n_outputs = N_CLASSES  # Binary classification
    
    # Create SepConv1D model directly
    model = SepConv1D(
        n_chans=n_channels,
        n_outputs=n_outputs,
        n_times=n_times,
        filters=32,          # Small filter count for small datasets
        kernel_size=16,      # Moderate temporal receptive field
        stride=8,            # Significant downsampling 
        padding=4,           # Preserve some temporal information
        dropout=0.2          # Light dropout for small datasets
    )
    
    print(f"Model created successfully!")
    print(f"Model parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy input
    batch_size = 16
    dummy_input = torch.randn(batch_size, n_channels, n_times).to(device)
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Output shape: {output.shape}")
    print(f"Forward pass successful!\n")

def demo_sepconv1d_with_config():
    """
    Demonstrate SepConv1D usage through config system
    """
    print("="*60) 
    print("SepConv1D Config System Usage Demo")
    print("="*60)
    
    # Show how to use SepConv1D through the create_model factory
    n_channels = 26
    
    print("Creating SepConv1D model through create_model factory...")
    
    model = create_model(
        n_channels=n_channels,
        is_lda=False,
        model_name='SepConv1D'
    )
    
    print(f"Model created: {type(model).__name__}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\nModel architecture summary:")
    print(f"  - Separable convolution with {SEPCONV1D_FILTERS} filters")
    print(f"  - Kernel size: {SEPCONV1D_KERNEL_SIZE}")
    print(f"  - Stride: {SEPCONV1D_STRIDE} (for parameter reduction)")
    print(f"  - Padding: {SEPCONV1D_PADDING}")
    print(f"  - Designed for small datasets with overfitting prevention")
    
def demo_small_dataset_protections():
    """
    Demonstrate automatic small dataset detection and protections
    """
    print("="*60)
    print("Small Dataset Protection Demo")
    print("="*60)
    
    # Import the detection function
    from models import detect_small_dataset
    
    # Create a mock small dataset loader
    class MockDataset:
        def __init__(self, size):
            self.size = size
        def __len__(self):
            return self.size
    
    class MockLoader:
        def __init__(self, dataset):
            self.dataset = dataset
            
    # Test with small dataset
    small_dataset = MockDataset(500)  # 500 samples < 1000 threshold
    small_loader = MockLoader(small_dataset)
    
    print("Testing with small dataset (500 samples):")
    config = detect_small_dataset(small_loader, 'SepConv1D')
    
    print(f"\nResulting configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
        
    # Test with normal dataset
    print("\n" + "-"*40)
    print("Testing with normal dataset (2000 samples):")
    normal_dataset = MockDataset(2000)  # 2000 samples > 1000 threshold  
    normal_loader = MockLoader(normal_dataset)
    
    config = detect_small_dataset(normal_loader, 'EEGConformer')
    
    print(f"\nResulting configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

def demo_fusion_compatibility():
    """
    Demonstrate SepConv1D compatibility with fusion methods
    """
    print("="*60)
    print("SepConv1D Fusion Compatibility Demo") 
    print("="*60)
    
    try:
        from models import create_fusion_model
        
        # Mock datasets info for fusion
        datasets_info = {
            'P3_dataset': {'channels': ['Fz', 'Cz', 'Pz', 'Oz']},
            'AVO_dataset': {'channels': ['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7']}
        }
        
        print("Testing SepConv1D with spatial attention fusion...")
        
        fusion_model = create_fusion_model(
            model_name='SepConv1D',
            datasets_info=datasets_info,
            fusion_method='spatial_attention',
            domain_adaptation='none'
        )
        
        print(f"Fusion model created successfully: {type(fusion_model).__name__}")
        
        total_params = sum(p.numel() for p in fusion_model.parameters())
        print(f"Total parameters: {total_params:,}")
        print("✓ SepConv1D is compatible with spatial attention fusion!")
        
    except Exception as e:
        print(f"Note: Fusion compatibility requires fusion_methods.py module")
        print(f"Error: {e}")

def print_usage_instructions():
    """
    Print instructions for using SepConv1D in practice
    """
    print("="*60)
    print("How to Use SepConv1D in Your Experiments")
    print("="*60)
    
    instructions = """
1. **Modify config.py to use SepConv1D:**
   
   # Change the classifier setting
   classifier = 'SepConv1D'
   
2. **SepConv1D is automatically optimized for small datasets:**
   - Automatically detects datasets < 1000 samples
   - Applies enhanced overfitting prevention measures
   - Uses lighter dropout (0.2 vs 0.3)
   - Stronger L2 regularization 
   - More aggressive early stopping

3. **Configure SepConv1D parameters (optional):**
   
   # In config.py, you can adjust:
   SEPCONV1D_FILTERS = 32        # Reduce for even smaller datasets
   SEPCONV1D_KERNEL_SIZE = 16    # Temporal receptive field
   SEPCONV1D_STRIDE = 8          # Higher stride = fewer parameters
   SEPCONV1D_PADDING = 4         # Temporal padding

4. **Use with fusion methods:**
   
   # In config.py:
   ELECTRODE_FUSION_METHOD = 'spatial_attention'
   # SepConv1D works with all fusion methods!

5. **Best practices for small datasets:**
   - Use cross-validation for robust evaluation
   - Consider data augmentation (already enabled)
   - Monitor training/validation curves for overfitting
   - SepConv1D's automatic protections will help

6. **Run your experiment:**
   
   conda activate eeg_realtime  # Activate environment
   python main.py               # Run experiment
   
   The system will automatically detect small datasets and apply
   appropriate overfitting prevention measures.
"""
    
    print(instructions)

if __name__ == "__main__":
    print("SepConv1D Integration Demo")
    print("=" * 80)
    
    # Run all demonstrations
    demo_sepconv1d_basic()
    print("\n")
    
    demo_sepconv1d_with_config()  
    print("\n")
    
    demo_small_dataset_protections()
    print("\n")
    
    demo_fusion_compatibility()
    print("\n")
    
    print_usage_instructions()
    
    print("\n" + "="*80)
    print("SepConv1D Integration Complete!")
    print("You can now use 'SepConv1D' as a classifier option in config.py")
    print("The model is specifically optimized for small datasets and overfitting prevention.")
    print("="*80)
