#!/usr/bin/env python3
"""
Test the unified electrode space implementation for graph_enhanced
"""

import torch
from fusion_methods import GraphEnhancedModel, FusionModelFactory
from models import SepConv1D
from constants import P3_CHANNELS, AVO_CHANNELS, COMMON_CHANNELS

def test_unified_electrode_space():
    """Test the unified electrode space and graph_enhanced model"""

    # Test parameters
    batch_size = 32
    n_timepoints = 128
    n_classes = 2

    # Mock datasets info with actual channel lists
    datasets_info = {
        'P3': {
            'channels': P3_CHANNELS,  # 30 channels
            'n_timepoints': n_timepoints
        },
        'AVO': {
            'channels': AVO_CHANNELS,  # 26 channels
            'n_timepoints': n_timepoints
        }
    }

    print("=== Testing Unified Electrode Space ===")
    print(f"P3 channels ({len(P3_CHANNELS)}): {P3_CHANNELS}")
    print(f"AVO channels ({len(AVO_CHANNELS)}): {AVO_CHANNELS}")
    print(f"Common channels ({len(COMMON_CHANNELS)}): {COMMON_CHANNELS}")

    # Create base model parameters for SepConv1D
    base_model_params = {
        'n_outputs': n_classes,
        'n_times': n_timepoints,
        'filters': 48,
        'kernel_size': 16,
        'stride': 8,
        'padding': 4,
        'dropout': 0.25
    }

    print("\n=== Creating Graph Enhanced Model ===")
    try:
        # Create hybrid model using the factory
        model = FusionModelFactory.create_fusion_model(
            fusion_method='graph_enhanced',
            datasets_info=datasets_info,
            base_model_info={
                'class': SepConv1D,
                'params': base_model_params
            }
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✓ Model created successfully")
        print(f"✓ Unified channels: {model.unified_n_channels}")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with P3 dataset data
    print(f"\n=== Testing P3 Dataset Forward Pass ===")
    x_p3 = torch.randn(batch_size, len(P3_CHANNELS), n_timepoints)
    print(f"Input shape: {x_p3.shape}")

    try:
        output_p3 = model(x_p3, dataset_name='P3')
        print(f"✓ P3 output shape: {output_p3.shape}")
        assert output_p3.shape == (batch_size, n_classes), f"Wrong output shape: {output_p3.shape}"
    except Exception as e:
        print(f"✗ P3 forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with AVO dataset data
    print(f"\n=== Testing AVO Dataset Forward Pass ===")
    x_avo = torch.randn(batch_size, len(AVO_CHANNELS), n_timepoints)
    print(f"Input shape: {x_avo.shape}")

    try:
        output_avo = model(x_avo, dataset_name='AVO')
        print(f"✓ AVO output shape: {output_avo.shape}")
        assert output_avo.shape == (batch_size, n_classes), f"Wrong output shape: {output_avo.shape}"
    except Exception as e:
        print(f"✗ AVO forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test without dataset_name (fallback mode)
    print(f"\n=== Testing Fallback Mode (no dataset_name) ===")
    try:
        output_fallback = model(x_p3)  # No dataset_name provided
        print(f"✓ Fallback output shape: {output_fallback.shape}")
        assert output_fallback.shape == (batch_size, n_classes), f"Wrong output shape: {output_fallback.shape}"
    except Exception as e:
        print(f"✗ Fallback mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test channel mapping information
    print(f"\n=== Channel Mapping Information ===")
    for dataset_name in ['P3', 'AVO']:
        mapping = model.channel_mapping[dataset_name]
        print(f"{dataset_name} channel mapping: {len(mapping)}/{len(datasets_info[dataset_name]['channels'])} channels mapped")
        print(f"  Sample mapping: {dict(list(mapping.items())[:5])}")  # Show first 5 mappings

    # Verify unified space contains key electrodes
    print(f"\n=== Unified Space Verification ===")
    key_electrodes = ['Fz', 'Cz', 'Pz', 'O1', 'O2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    present_electrodes = [ch for ch in key_electrodes if ch in model.unified_channels]
    print(f"Key electrodes present: {len(present_electrodes)}/{len(key_electrodes)}")
    print(f"Present: {present_electrodes}")
    missing_electrodes = [ch for ch in key_electrodes if ch not in model.unified_channels]
    if missing_electrodes:
        print(f"Missing: {missing_electrodes}")

    return True

def compare_with_baseline():
    """Compare unified channels with baseline approach"""

    print(f"\n=== Comparison with Baseline ===")
    print(f"Baseline approach:")
    print(f"  - P3 only: {len(P3_CHANNELS)} channels")
    print(f"  - AVO only: {len(AVO_CHANNELS)} channels")
    print(f"  - Common only: {len(COMMON_CHANNELS)} channels")
    print(f"  - Max channels: {max(len(P3_CHANNELS), len(AVO_CHANNELS))} channels")

    # Test what our unified space gives us
    datasets_info = {
        'P3': {'channels': P3_CHANNELS},
        'AVO': {'channels': AVO_CHANNELS}
    }

    from electrode_utils import create_unified_electrode_space
    unified_channels, channel_mapping = create_unified_electrode_space(datasets_info)

    print(f"\nUnified approach:")
    print(f"  - Unified space: {len(unified_channels)} channels")
    print(f"  - P3 mapped: {len(channel_mapping['P3'])}/{len(P3_CHANNELS)} channels")
    print(f"  - AVO mapped: {len(channel_mapping['AVO'])}/{len(AVO_CHANNELS)} channels")

    # Coverage analysis
    p3_coverage = len(channel_mapping['P3']) / len(P3_CHANNELS) * 100
    avo_coverage = len(channel_mapping['AVO']) / len(AVO_CHANNELS) * 100

    print(f"\nCoverage analysis:")
    print(f"  - P3 coverage: {p3_coverage:.1f}%")
    print(f"  - AVO coverage: {avo_coverage:.1f}%")

if __name__ == "__main__":
    print("Testing Unified Electrode Space for Graph Enhanced Model")
    print("=" * 60)

    success = test_unified_electrode_space()

    if success:
        compare_with_baseline()

    print("=" * 60)
    if success:
        print("✅ All tests passed! Unified electrode space works correctly.")
    else:
        print("❌ Tests failed. Please check the implementation.")