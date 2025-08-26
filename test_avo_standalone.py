"""
Standalone test for AVO dataset using the actual main.py workflow
"""

import os
import sys

def test_avo_main():
    """Test AVO dataset with the actual main.py workflow."""
    
    print("="*60)
    print("TESTING AVO WITH MAIN.PY WORKFLOW")
    print("="*60)
    
    # Set environment variables to test AVO dataset
    os.environ['DATASET'] = 'ds005863'
    os.environ['DATA_DIR'] = '../ds005863'
    os.environ['USE_COMBINED_DATASETS'] = '0'
    os.environ['ELECTRODE_LIST'] = 'common'
    os.environ['CLASSIFIER'] = 'lda'
    os.environ['SEPARATE_SUBJECT_CLASSIFICATION'] = 'True'
    os.environ['SEEDS'] = '1'
    
    print("Environment variables set for AVO dataset test:")
    print(f"  DATASET: {os.environ['DATASET']}")
    print(f"  DATA_DIR: {os.environ['DATA_DIR']}")
    print(f"  CLASSIFIER: {os.environ['CLASSIFIER']}")
    print(f"  SEPARATE_SUBJECT_CLASSIFICATION: {os.environ['SEPARATE_SUBJECT_CLASSIFICATION']}")
    
    print("\nRunning main.py with AVO dataset...")
    print("This will test if the trial count fix works in the actual workflow.")
    print("Look for trial counts around 200 instead of 400.")
    
    try:
        # Import and run main
        import main
        main.main()
        
        print("\n✅ AVO test completed successfully!")
        print("Check the output above for trial counts - they should be ~200 per subject.")
        
    except Exception as e:
        print(f"\n❌ Error running AVO test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_avo_main()
