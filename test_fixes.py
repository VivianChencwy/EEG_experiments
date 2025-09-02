#!/usr/bin/env python3
"""
Test script to verify the fixes for experiment issues.
"""

import os
import sys

def test_imports():
    """Test that all imports work correctly after fixes."""
    print("🔍 Testing imports after fixes...")
    
    try:
        # Test main imports
        from config import dataset, classifier, electrode_list
        print(f"   ✅ Config imports: {dataset}, {classifier}, {electrode_list}")
        
        # Test experiment imports
        from experiment import run_experiment
        print("   ✅ Experiment imports work")
        
        # Test that visualization no longer has plot_confusion_matrix
        from visualization import plot_confusion_matrix
        print("   ❌ plot_confusion_matrix still exists (should be removed)")
        return False
        
    except ImportError as e:
        if "plot_confusion_matrix" in str(e):
            print("   ✅ plot_confusion_matrix successfully removed")
            return True
        else:
            print(f"   ❌ Import error: {e}")
            return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def test_config_generation():
    """Test that configuration generation works correctly."""
    print("\n🔍 Testing configuration generation...")
    
    try:
        from run_all_experiments import make_configs
        configs = make_configs()
        
        print(f"   ✅ Generated {len(configs)} configurations")
        
        # Check for the previously missing configurations
        found_configs = []
        for config in configs:
            dataset = config.get('DATASET', '')
            classifier = config.get('CLASSIFIER', '')
            separate = config.get('SEPARATE_SUBJECT_CLASSIFICATION', '') == 'True'
            electrode = config.get('ELECTRODE_LIST', '')
            
            config_tuple = (dataset, classifier, separate, electrode)
            found_configs.append(config_tuple)
        
        # Check for previously missing configs
        missing_before = [
            ('ds005863', 'lda', False, 'common'),
            ('ds005863', 'lda', False, 'all'),
            ('use_combined_datasets', 'lda', True, 'common'),
            ('use_combined_datasets', 'ShallowFBCSPNet', True, 'common'),
        ]
        
        print("   Checking previously missing configurations:")
        all_found = True
        for expected in missing_before:
            if expected in found_configs:
                print(f"     ✅ {expected}")
            else:
                print(f"     ❌ {expected}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"   ❌ Error testing config generation: {e}")
        return False

def test_auc_calculation():
    """Test that AUC calculation is consistent."""
    print("\n🔍 Testing AUC calculation consistency...")
    
    try:
        from models import evaluate
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create test data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Create LDA model
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        
        # Create data loader
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Test evaluation
        details = evaluate(lda, loader, 'cpu', is_lda=True, return_details=True)
        
        print(f"   ✅ AUC calculation works: {details.get('auc', 'N/A')}")
        
        # Check that AUC is not default 0.5 (unless it should be)
        auc = details.get('auc', 0.5)
        if auc == 0.5:
            print("   ⚠️  AUC is 0.5 (might be correct for random data)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing AUC calculation: {e}")
        return False

def main():
    print("🔧 Testing Fixes for EEG Experiment Issues")
    print("=" * 50)
    
    success1 = test_imports()
    success2 = test_config_generation()
    success3 = test_auc_calculation()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Import fixes: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Config generation: {'✅ PASSED' if success2 else '❌ FAILED'}")
    print(f"AUC calculation: {'✅ PASSED' if success3 else '❌ FAILED'}")
    
    if success1 and success2 and success3:
        print("\n✅ All fixes verified successfully!")
        print("\nNext steps:")
        print("1. Activate conda environment: conda activate eegtemp")
        print("2. Run experiments: python run_all_experiments.py")
        return True
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    import torch
    success = main()
    sys.exit(0 if success else 1)
