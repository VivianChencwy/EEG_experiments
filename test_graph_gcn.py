#!/usr/bin/env python3
"""
Test the graph_gcn fusion model to ensure proper complexity
"""

import torch
import time
from fusion_methods import UniversalFeatureSpace

def test_graph_gcn_complexity():
    """Test that graph_gcn has appropriate computational complexity"""

    # Test parameters
    batch_size = 32
    n_timepoints = 128
    n_classes = 2

    # Mock datasets info
    datasets_info = {
        'P3': {
            'channels': [f'C{i}' for i in range(30)],  # 30 channels
            'n_timepoints': n_timepoints
        },
        'AVO': {
            'channels': [f'C{i}' for i in range(26)],  # 26 channels
            'n_timepoints': n_timepoints
        }
    }

    print("=== Testing Graph GCN Model Complexity ===")

    # Create model
    model = UniversalFeatureSpace(datasets_info, n_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with P3 dataset (30 channels)
    print(f"\n=== Testing P3 Dataset (30 channels) ===")
    x_p3 = torch.randn(batch_size, 30, n_timepoints)

    # Warm up
    with torch.no_grad():
        _ = model(x_p3, 'P3')

    # Time forward pass
    start_time = time.time()
    num_runs = 100
    for _ in range(num_runs):
        output = model(x_p3, 'P3')
    forward_time = (time.time() - start_time) / num_runs

    print(f"✓ P3 input shape: {x_p3.shape}")
    print(f"✓ P3 output shape: {output.shape}")
    print(f"✓ Average forward time: {forward_time*1000:.2f}ms")

    # Test with AVO dataset (26 channels)
    print(f"\n=== Testing AVO Dataset (26 channels) ===")
    x_avo = torch.randn(batch_size, 26, n_timepoints)

    # Warm up
    with torch.no_grad():
        _ = model(x_avo, 'AVO')

    # Time forward pass
    start_time = time.time()
    for _ in range(num_runs):
        output = model(x_avo, 'AVO')
    forward_time = (time.time() - start_time) / num_runs

    print(f"✓ AVO input shape: {x_avo.shape}")
    print(f"✓ AVO output shape: {output.shape}")
    print(f"✓ Average forward time: {forward_time*1000:.2f}ms")

    # Test gradient computation (essential for training)
    print(f"\n=== Testing Gradient Computation ===")
    model.train()
    x_test = torch.randn(batch_size, 30, n_timepoints, requires_grad=True)
    output = model(x_test, 'P3')
    loss = output.sum()

    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time

    print(f"✓ Backward pass time: {backward_time*1000:.2f}ms")

    # Check that gradients exist
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1

    print(f"✓ Parameters with gradients: {grad_count}/{len(list(model.parameters()))}")

    # Test model memory usage
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"✓ Model memory: {model_size:.2f} MB")

    # Complexity verification
    expected_min_time = 1.0  # Expect at least 1ms per forward pass for a complex model
    if forward_time * 1000 < expected_min_time:
        print(f"⚠️  WARNING: Forward time ({forward_time*1000:.2f}ms) is suspiciously fast!")
        print("   This may indicate the model is not complex enough.")
        return False

    if trainable_params < 10000:
        print(f"⚠️  WARNING: Too few trainable parameters ({trainable_params:,})")
        print("   Model may be too simple.")
        return False

    print(f"\n✅ Graph GCN complexity test PASSED")
    print(f"   - Sufficient parameters: {trainable_params:,}")
    print(f"   - Reasonable forward time: {forward_time*1000:.2f}ms")
    print(f"   - Proper gradients: {grad_count} parameters")

    return True

if __name__ == "__main__":
    print("Testing Graph GCN Model Complexity")
    print("=" * 50)

    success = test_graph_gcn_complexity()

    print("=" * 50)
    if success:
        print("✅ All tests passed! Graph GCN has appropriate complexity.")
    else:
        print("❌ Tests failed. Graph GCN may have complexity issues.")