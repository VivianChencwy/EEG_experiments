"""
Simple runner script for improved experiments
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_experiment_runner import main

if __name__ == "__main__":
    print("Starting Improved EEG Experiments...")
    results = main()
    
    if results:
        print("\nExperiment completed successfully!")
        
        # Print final summary
        print("\nFINAL RESULTS SUMMARY:")
        print("-" * 50)
        for model_name, result in results.items():
            if result is not None:
                if isinstance(result, dict):
                    print(f"{model_name}: {result['accuracy']:.1%} accuracy")
                else:
                    print(f"{model_name}: {result:.1%} accuracy")
            else:
                print(f"{model_name}: Failed")
    else:
        print("\nExperiment failed!")
        sys.exit(1)