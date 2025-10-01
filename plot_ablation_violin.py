import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")

# Define file paths for AS-MMD and ablation experiments
# NOTE: Both avo_tfdwt and p3_tfdwt files have same data (P3 as target, P3=10 AVO=80)
# Need to run TF-DWT with P3=80 AVO=10 for AVO as target comparison
data_files = {
    # 'AVO (Dataset 1)': {
    #     'AS-MMD (Proposed)': '0930/MISSING_avo_target_tfdwt.csv',  # Need to run: P3=80,AVO=10
    #     'Ablation 1: Equal Weights': 'ablation_results_AVOsmall/ablation1_equal_weights_detailed_PENDING.csv',
    #     'Ablation 2: Fixed Weights': 'ablation_results_AVOsmall/ablation2_fixed_weights_detailed_PENDING.csv',
    #     'Ablation 3: No MMD': 'ablation_results_AVOsmall/ablation3_no_mmd_detailed_PENDING.csv',
    #     'Ablation 4: No Split BN': 'ablation_results_AVOsmall/ablation4_no_split_bn_detailed_PENDING.csv'
    # },
    'P3 (Dataset 2)': {
        'AS-MMD (Proposed)': '0930/p3_tfdwt_detailed_results_20250929_160212.csv',  # P3=10,AVO=80: P3 is target
        'Ablation 1: Equal Weights': 'ablation_results_P3small/ablation1_equal_weights_detailed_20250930_191928.csv',
        'Ablation 2: Fixed Weights': 'ablation_results_P3small/ablation2_fixed_weights_detailed_20250930_192607.csv',
        'Ablation 3: No MMD': 'ablation_results_P3small/ablation3_no_mmd_detailed_20250930_193652.csv',
        'Ablation 4: No Split BN': 'ablation_results_P3small/ablation4_no_split_bn_detailed_20250930_194307.csv'
    }
}

# Prepare data
def prepare_data():
    all_data = []
    for dataset, methods in data_files.items():
        for method, file_path in methods.items():
            df = pd.read_csv(file_path)

            # Determine which dataset we're looking at (target dataset for small dataset experiments)
            if 'Dataset 1' in dataset:
                acc_col, auc_col = 'avo_accuracy', 'avo_auc'
            else:
                acc_col, auc_col = 'p3_accuracy', 'p3_auc'

            for _, row in df.iterrows():
                all_data.append({
                    'Dataset': dataset,
                    'Method': method,
                    'Accuracy': row[acc_col],
                    'AUC': row[auc_col]
                })
    return pd.DataFrame(all_data)

# Create violin plots
def create_violin_plots():
    data = prepare_data()

    method_order = [
        'AS-MMD (Proposed)',
        'Ablation 1: Equal Weights',
        'Ablation 2: Fixed Weights',
        'Ablation 3: No MMD',
        'Ablation 4: No Split BN'
    ]

    # Define colors for methods
    colors = ['#80b1d3', '#fdb462', '#fb8072', '#bebada', '#8dd3c7']
    palette = dict(zip(method_order, colors))

    # Calculate y-limits
    acc_min, acc_max = data['Accuracy'].min(), data['Accuracy'].max()
    acc_range = acc_max - acc_min
    acc_ylim = [max(0, acc_min - 0.15 * acc_range), min(1, acc_max + 0.15 * acc_range)]

    auc_min, auc_max = data['AUC'].min(), data['AUC'].max()
    auc_range = auc_max - auc_min
    auc_ylim = [max(0, auc_min - 0.15 * auc_range), min(1, auc_max + 0.15 * auc_range)]

    # ---------------- Accuracy ----------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=data, x='Dataset', y='Accuracy', hue='Method',
        hue_order=method_order, palette=palette, ax=ax1,
        inner='box', cut=0, density_norm='width'
    )

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:5], labels[:5], title='Method',
               fontsize=10, title_fontsize=11, loc='lower right')

    ax1.set_title('Accuracy Distribution: AS-MMD vs Ablation Studies', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Dataset', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(acc_ylim)
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig('0930/ablation_violin_plot_accuracy.png', dpi=300, bbox_inches='tight')
    plt.savefig('0930/ablation_violin_plot_accuracy.pdf', bbox_inches='tight')
    plt.close()

    # ---------------- AUC ----------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=data, x='Dataset', y='AUC', hue='Method',
        hue_order=method_order, palette=palette, ax=ax2,
        inner='box', cut=0, density_norm='width'
    )

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:5], labels[:5], title='Method',
               fontsize=10, title_fontsize=11, loc='lower right')

    ax2.set_title('AUC Distribution: AS-MMD vs Ablation Studies', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Dataset', fontsize=14)
    ax2.set_ylabel('AUC', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(auc_ylim)
    ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig('0930/ablation_violin_plot_auc.png', dpi=300, bbox_inches='tight')
    plt.savefig('0930/ablation_violin_plot_auc.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    create_violin_plots()
    print("Ablation violin plots saved to 0930/ directory")
