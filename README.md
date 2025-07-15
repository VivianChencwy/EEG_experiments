# EEG Oddball Classification Experiments

A modular Python framework for EEG classification experiments using P3 and Active Visual Oddball (AVO) datasets. This codebase supports both individual subject training and cross-dataset generalization experiments with multiple neural network architectures.

## Installation

1. **Clone the repository**:
   
   ```bash
   git clone <repository-url>
   cd EEG_experiments
   ```
   
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up data paths** in `config.py`:
   
   ```python
   P3_DATA_DIR = "path/to/your/P3/data"
   AVO_DATA_DIR = "path/to/your/AVO/data"
   ```

## Quick Start

### Basic Usage

```bash
python main.py
```

The experiment configuration is controlled through `config.py`. Here are the key parameters:

### Configuration Options

#### 1. Dataset Selection
```python
# Single dataset experiments
data_dir = P3_DATA_DIR  
dataset = 'P3 Raw Data BIDS-Compatible'
use_combined_datasets = False

# Combined dataset experiments  
use_combined_datasets = True  # Uses both P3 and AVO
```

#### 2. Training Strategy
```python
# Pooled training: All subjects' data combined into one model
separate_subject_classification = False

# Individual training: Each subject gets their own model  
separate_subject_classification = True
```

#### 3. Electrode Configuration
```python
# Use common electrodes across datasets (recommended for combined experiments)
electrode_list = 'common'  

# Use all available electrodes for single dataset
electrode_list = 'all'
```

#### 4. Classifier Selection
```python
# Deep learning approach (recommended)
classifier = 'ShallowFBCSPNet'

# Traditional machine learning
classifier = 'lda'  # Linear Discriminant Analysis
```

#### 5. Custom Hyperparameters

```python
# Training configuration
BATCH_SIZE = 64
MAX_EPOCHS = 100  
LEARNING_RATE = 0.001
TRAIN_SIZE = 0.6 
VAL_SIZE = 0.2    
TEST_SIZE = 0.2   

# Reproducibility
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Multiple seeds for robust results
```

### Log Files

Detailed logs are saved in the `./log/` directory with timestamps and configuration:
```
log/Combined_clf-ShallowFBCSPNet_sep-False_el-common_results_20241201_143022.log
```

### Adding New Datasets

1. **Add preprocessing logic** in `preprocessor.py`
2. **Update constants** in `constants.py` 
3. **Modify experiment logic** in `experiment.py`
4. **Update configuration** in `config.py`

### Adding New Models

1. **Add model creation** in `models.py`:

   ```python
   def create_model(n_channels, is_lda=False):
       if is_lda:
           return LinearDiscriminantAnalysis()
       # Add your custom model here
       return YourCustomModel(n_channels)
   ```

2. **Update config.py**:

   ```python
   classifier = 'YourCustomModel'
   ```

## Code Structure

```
├── config.py           # All experiment parameters and hyperparameters
├── constants.py        # Channel names and event code definitions  
├── preprocessor.py     # OddballPreprocessor class for EEG data processing
├── models.py          # Model creation, training, and evaluation functions
├── utils.py           # Data loading, statistics, and utility functions
├── experiment.py      # Core experiment logic and training procedures
├── main.py            # Main entry point and experiment orchestration
├── experiment_logger.py # Logging and result tracking utilities
└── requirements.txt   # Python package dependencies
```

