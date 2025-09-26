#!/usr/bin/env python3
"""
Debug TF-DWT Parameter Modifications

This script only tests that parameter modifications work without running full experiments.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def test_parameter_modifications():
    """Test that our parameter modification logic works correctly"""

    print("🔧 Testing TF-DWT Parameter Modifications")

    # Read original main_tfdwt.py
    tfdwt_path = PROJECT_ROOT / 'main_tfdwt.py'
    original_content = tfdwt_path.read_text(encoding='utf-8')

    # Test parameters
    test_params = {
        'w_small_cap': 2.5,
        'mmd_thresholds': (1.5, 3.0, 0.05, 0.15, 0.25),
        'guard_factors': (0.7, 0.4, 0.6),
        'warmup_config': (3, 7, 0.12),
    }

    print(f"Original parameters found in code:")

    # Check original values
    if 'w_small_target = min(w_small_target, 3.0)' in original_content:
        print("✅ Found w_small_cap line (currently 3.0)")
    else:
        print("❌ w_small_cap line not found")

    if 'lambda_mmd = 0.1 if overall_ratio < 2.0 else (0.2 if overall_ratio < 4.0 else 0.3)' in original_content:
        print("✅ Found MMD lambda logic")
    else:
        print("❌ MMD lambda logic not found")

    if 'new_w = max(1.0, cur_w * 0.8)' in original_content:
        print("✅ Found guard factor for small domain weight")
    else:
        print("❌ Guard factor for weight not found")

    if 'warmup = max(2, min(5, int(0.1 * MAX_EPOCHS)))' in original_content:
        print("✅ Found warmup configuration")
    else:
        print("❌ Warmup configuration not found")

    print(f"\nApplying test modifications...")

    # Apply modifications
    modified_content = original_content

    # 1. w_small_cap
    w_cap = test_params['w_small_cap']
    modified_content = modified_content.replace(
        'w_small_target = min(w_small_target, 3.0)',
        f'w_small_target = min(w_small_target, {w_cap})'
    )
    print(f"✅ Modified w_small_cap to {w_cap}")

    # 2. MMD thresholds
    thresh1, thresh2, lambda1, lambda2, lambda3 = test_params['mmd_thresholds']
    old_lambda = 'lambda_mmd = 0.1 if overall_ratio < 2.0 else (0.2 if overall_ratio < 4.0 else 0.3)'
    new_lambda = f'lambda_mmd = {lambda1} if overall_ratio < {thresh1} else ({lambda2} if overall_ratio < {thresh2} else {lambda3})'
    modified_content = modified_content.replace(old_lambda, new_lambda)
    print(f"✅ Modified MMD lambda logic")

    # 3. Guard factors
    small_w_decay, small_mmd_decay, large_mmd_decay = test_params['guard_factors']

    # Small weight decay
    if 'new_w = max(1.0, cur_w * 0.8)' in modified_content:
        modified_content = modified_content.replace(
            'new_w = max(1.0, cur_w * 0.8)',
            f'new_w = max(1.0, cur_w * {small_w_decay})'
        )
        print(f"✅ Modified small domain weight decay to {small_w_decay}")

    # Small MMD decay
    if 'new_lambda = max(0.0, cur_lambda * 0.5)' in modified_content:
        modified_content = modified_content.replace(
            'new_lambda = max(0.0, cur_lambda * 0.5)',
            f'new_lambda = max(0.0, cur_lambda * {small_mmd_decay})'
        )
        print(f"✅ Modified small domain MMD decay to {small_mmd_decay}")

    # Large MMD decay
    if 'new_lambda = max(0.0, cur_lambda * 0.7)' in modified_content:
        modified_content = modified_content.replace(
            'new_lambda = max(0.0, cur_lambda * 0.7)',
            f'new_lambda = max(0.0, cur_lambda * {large_mmd_decay})'
        )
        print(f"✅ Modified large domain MMD decay to {large_mmd_decay}")

    # 4. Warmup config
    min_epochs, max_epochs, ratio = test_params['warmup_config']
    old_warmup = 'warmup = max(2, min(5, int(0.1 * MAX_EPOCHS)))'
    new_warmup = f'warmup = max({min_epochs}, min({max_epochs}, int({ratio} * MAX_EPOCHS)))'
    if old_warmup in modified_content:
        modified_content = modified_content.replace(old_warmup, new_warmup)
        print(f"✅ Modified warmup config to ({min_epochs}, {max_epochs}, {ratio})")

    # Verify modifications
    print(f"\n🔍 Verifying modifications:")

    changes_verified = 0
    total_changes = 4

    if f'w_small_target = min(w_small_target, {w_cap})' in modified_content:
        print(f"✅ w_small_cap verified: {w_cap}")
        changes_verified += 1
    else:
        print(f"❌ w_small_cap verification failed")

    if new_lambda in modified_content:
        print(f"✅ MMD lambda verified")
        changes_verified += 1
    else:
        print(f"❌ MMD lambda verification failed")

    if f'cur_w * {small_w_decay}' in modified_content:
        print(f"✅ Guard factors verified")
        changes_verified += 1
    else:
        print(f"❌ Guard factors verification failed")

    if new_warmup in modified_content:
        print(f"✅ Warmup config verified")
        changes_verified += 1
    else:
        print(f"❌ Warmup config verification failed")

    print(f"\n📊 Summary: {changes_verified}/{total_changes} modifications successful")

    if changes_verified == total_changes:
        print("🎉 All parameter modifications working correctly!")

        # Save a sample modified file for inspection
        sample_file = PROJECT_ROOT / 'main_tfdwt_modified_sample.py'
        with open(sample_file, 'w') as f:
            f.write(modified_content)
        print(f"📁 Sample modified file saved to: {sample_file}")

        return True
    else:
        print("⚠️  Some modifications failed - check the logic")
        return False

def test_config_modifications():
    """Test configuration modification logic"""
    print(f"\n🔧 Testing Config Modifications")

    # Read base config
    config_path = PROJECT_ROOT / 'config.py'
    original_config = config_path.read_text(encoding='utf-8')

    test_overrides = {
        'use_combined_datasets': True,
        'NESTED_CV_OUTER_FOLDS': 3,
        'NESTED_CV_REPEATS': 2,
        'EARLY_STOPPING_PATIENCE': 30,
        'LEARNING_RATE': 0.012,
    }

    # Apply modifications
    import re
    lines = original_config.splitlines()

    def set_line(prefix: str, value_src: str):
        nonlocal lines
        pat = re.compile(rf"^({re.escape(prefix)}\s*=).*$")
        replaced = False
        for i, line in enumerate(lines):
            if pat.match(line.strip()):
                lines[i] = f"{prefix} = {value_src}"
                replaced = True
                print(f"✅ Modified {prefix} = {value_src}")
                break
        if not replaced:
            lines.append(f"{prefix} = {value_src}")
            print(f"➕ Added {prefix} = {value_src}")

    for key, value in test_overrides.items():
        if key == 'use_combined_datasets':
            set_line(key, 'True' if value else 'False')
        else:
            set_line(key, str(value))

    modified_config = "\n".join(lines) + "\n"

    # Verify
    changes_found = 0
    for key, value in test_overrides.items():
        if key == 'use_combined_datasets':
            search_str = f"{key} = {'True' if value else 'False'}"
        else:
            search_str = f"{key} = {value}"

        if search_str in modified_config:
            print(f"✅ Verified: {search_str}")
            changes_found += 1
        else:
            print(f"❌ Failed: {search_str}")

    print(f"📊 Config modifications: {changes_found}/{len(test_overrides)} successful")
    return changes_found == len(test_overrides)

def main():
    """Run all parameter modification tests"""
    print("🧪 TF-DWT Parameter Modification Test Suite")

    # Test 1: TF-DWT parameter modifications
    tfdwt_success = test_parameter_modifications()

    # Test 2: Config modifications
    config_success = test_config_modifications()

    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")

    print(f"TF-DWT parameter modifications: {'✅ PASS' if tfdwt_success else '❌ FAIL'}")
    print(f"Config modifications: {'✅ PASS' if config_success else '❌ FAIL'}")

    if tfdwt_success and config_success:
        print("\n🎉 All tests passed! Parameter optimization should work correctly.")
        print("\n📋 Next steps:")
        print("1. Run quick_tfdwt_test.py for actual experiments")
        print("2. Run tfdwt_param_optimizer.py for full optimization")
        print("3. Use run_tfdwt_optimization.py for complete process")
    else:
        print("\n⚠️  Some tests failed. Fix the issues before running full optimization.")

    return tfdwt_success and config_success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)