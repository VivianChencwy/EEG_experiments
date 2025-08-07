#!/usr/bin/env python3
"""
Test script to verify subject layer functionality and backward compatibility
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Test imports
try:
    from models import create_model, SubjectInputLayer, ShallowFBCSPNetWithSubjectLayer
    from experiment import SubjectDataset
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

def test_subject_layer():
    """Test SubjectInputLayer functionality"""
    print("\n=== Testing SubjectInputLayer ===")
    
    # Test parameters
    n_subjects = 5
    n_channels = 16
    n_timepoints = 128
    batch_size = 4
    
    # Create subject layer
    subject_layer = SubjectInputLayer(n_subjects, n_channels)
    print(f"✓ Created SubjectInputLayer with {n_subjects} subjects, {n_channels} channels")
    
    # Test input data
    x = torch.randn(batch_size, n_channels, n_timepoints)
    subject_indices = torch.randint(0, n_subjects, (batch_size,))
    
    # Forward pass
    output = subject_layer(x, subject_indices)
    
    # Verify output shape
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")
    
    # Test with different subject indices
    subject_indices_all_zero = torch.zeros(batch_size, dtype=torch.long)
    output_zero = subject_layer(x, subject_indices_all_zero)
    assert output_zero.shape == x.shape
    print("✓ Forward pass with zero indices successful")

def test_model_creation():
    """Test model creation with and without subject layer"""
    print("\n=== Testing Model Creation ===")
    
    n_channels = 16
    n_subjects = 5
    
    # Test standard model creation
    model_standard = create_model(n_channels, is_lda=False)
    print(f"✓ Created standard model: {type(model_standard).__name__}")
    
    # Test LDA model creation
    model_lda = create_model(n_channels, is_lda=True)
    print(f"✓ Created LDA model: {type(model_lda).__name__}")
    
    # Test model with subject layer disabled
    model_no_subj = create_model(n_channels, is_lda=False, enable_subject_layer=False)
    print(f"✓ Created model without subject layer: {type(model_no_subj).__name__}")
    
    # Test model with subject layer enabled
    model_with_subj = create_model(
        n_channels, 
        is_lda=False, 
        n_subjects=n_subjects, 
        enable_subject_layer=True
    )
    print(f"✓ Created model with subject layer: {type(model_with_subj).__name__}")
    
    # Test forward pass with subject layer model
    x = torch.randn(2, n_channels, 128)
    subject_indices = torch.randint(0, n_subjects, (2,))
    
    # Standard model (should work without subject indices)
    output_standard = model_standard(x)
    print(f"✓ Standard model forward pass: {x.shape} -> {output_standard.shape}")
    
    # Model with subject layer
    output_with_subj = model_with_subj(x, subject_indices)
    print(f"✓ Subject layer model forward pass: {x.shape} -> {output_with_subj.shape}")

def test_subject_dataset():
    """Test SubjectDataset functionality"""
    print("\n=== Testing SubjectDataset ===")
    
    # Create test data
    n_samples = 10
    n_channels = 16
    n_timepoints = 128
    
    data = torch.randn(n_samples, n_channels, n_timepoints)
    labels = torch.randint(0, 2, (n_samples,))
    subject_indices = torch.randint(0, 3, (n_samples,))
    
    # Create dataset
    dataset = SubjectDataset(data, labels, subject_indices)
    print(f"✓ Created SubjectDataset with {len(dataset)} samples")
    
    # Test indexing
    sample_data, sample_label, sample_subject = dataset[0]
    assert sample_data.shape == data[0].shape
    assert sample_label == labels[0]
    assert sample_subject == subject_indices[0]
    print("✓ Dataset indexing works correctly")
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    batch_data, batch_labels, batch_subjects = next(iter(loader))
    assert batch_data.shape[0] == 3
    assert batch_labels.shape[0] == 3
    assert batch_subjects.shape[0] == 3
    print("✓ DataLoader works correctly")

def test_backward_compatibility():
    """Test that existing functionality still works"""
    print("\n=== Testing Backward Compatibility ===")
    
    # Test standard TensorDataset still works
    n_samples = 5
    n_channels = 16
    n_timepoints = 128
    
    data = torch.randn(n_samples, n_channels, n_timepoints)
    labels = torch.randint(0, 2, (n_samples,))
    
    # Standard dataset (should still work)
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=2)
    
    batch_data, batch_labels = next(iter(loader))
    assert len(batch_data.shape) == 3  # Should be (batch, channels, time)
    assert len(batch_labels.shape) == 1  # Should be (batch,)
    print("✓ Standard TensorDataset compatibility maintained")
    
    # Test model creation without optional parameters
    model = create_model(n_channels)
    output = model(data)
    print(f"✓ Model creation and forward pass without optional params: {data.shape} -> {output.shape}")

if __name__ == "__main__":
    print("Starting subject layer tests...")
    
    try:
        test_subject_layer()
        test_model_creation()
        test_subject_dataset()
        test_backward_compatibility()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED!")
        print("✓ Subject layer functionality working")
        print("✓ Backward compatibility maintained")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)