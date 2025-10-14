#!/usr/bin/env python3
"""
Test the graph enhanced hybrid model
"""

import torch
from fusion_methods import LightweightGraphEnhancer, GraphEnhancedModel
from models import SepConv1D

def test_graph_enhanced_model():
    """Test the hybrid model with correct dimensions"""

    # Test parameters - using actual dimensions from error log
    batch_size = 32
    n_channels = 30  # Actual max channels
    n_timepoints = 128
    n_classes = 2

    # Create test data
    x = torch.randn(batch_size, n_channels, n_timepoints)

    print(f"Test input shape: {x.shape}")

    # Test LightweightGraphEnhancer independently
    print("=== Testing LightweightGraphEnhancer ===")
    enhancer = LightweightGraphEnhancer(n_channels=n_channels)

    try:
        enhanced_x = enhancer(x)  # No channels provided, should use simple adjacency
        print(f"✓ Enhancer output shape: {enhanced_x.shape}")
        assert enhanced_x.shape == x.shape, f"Shape mismatch: {enhanced_x.shape} vs {x.shape}"
        print("✓ LightweightGraphEnhancer works correctly")
    except Exception as e:
        print(f"✗ LightweightGraphEnhancer failed: {e}")
        return False

    # Test GraphEnhancedModel
    print("\n=== Testing GraphEnhancedModel ===")

    # Create base model parameters for SepConv1D
    base_model_params = {
        'n_chans': n_channels,
        'n_outputs': n_classes,
        'n_times': n_timepoints,
        'filters': 48,
        'kernel_size': 16,
        'stride': 8,
        'padding': 4,
        'dropout': 0.25
    }

    try:
        # Create hybrid model
        hybrid_model = GraphEnhancedModel(
            base_model_class=SepConv1D,
            base_model_params=base_model_params,
            n_channels=n_channels,
            enhancement_strength=0.1
        )

        # Test forward pass
        output = hybrid_model(x)
        print(f"✓ Hybrid model output shape: {output.shape}")
        assert output.shape == (batch_size, n_classes), f"Output shape mismatch: {output.shape}"
        print("✓ GraphEnhancedModel works correctly")

        # Test with different input sizes
        print("\n=== Testing with different input sizes ===")
        for test_channels in [26, 30, 32]:
            test_x = torch.randn(batch_size, test_channels, n_timepoints)
            test_output = hybrid_model(test_x)
            print(f"✓ Input: {test_x.shape} -> Output: {test_output.shape}")

        return True

    except Exception as e:
        print(f"✗ GraphEnhancedModel failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Graph Enhanced Hybrid Model")
    print("=" * 50)

    success = test_graph_enhanced_model()

    print("=" * 50)
    if success:
        print("✓ All tests passed! Graph enhanced model is ready.")
    else:
        print("✗ Tests failed. Please check the implementation.")