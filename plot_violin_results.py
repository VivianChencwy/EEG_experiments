import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")

# Define file paths and method names
data_files = {
    'AVO': {
        'Single-dataset training': '0930/main_AVO_detailed_results_20250928_183126.csv',
        'Combined training': '0930/avo_main_combined_detailed_results_20250928_184539.csv',
        'AS-MMD (Proposed)': '0930/avo_tfdwt_detailed_results_20250929_172704.csv'
    },
    'P3': {
        'Single-dataset training': '0930/main_P3_detailed_results_20250930_190550.csv',
        'Combined training': '0930/p3_main_combined_detailed_results_20250930_185358.csv',
        'AS-MMD (Proposed)': '0930/p3_tfdwt_detailed_results_20250929_160212.csv'
    }
}

# Prepare data (25个点直接使用，不再分组均值)
def prepare_data():
    all_data = []
    for dataset, methods in data_files.items():
        for method, file_path in methods.items():
            df = pd.read_csv(file_path)

            if dataset == 'AVO':
                acc_col, auc_col = 'avo_accuracy', 'avo_auc'
            else:
                acc_col, auc_col = 'p3_accuracy', 'p3_auc'

            for _, row in df.iterrows():
                all_data.append({
                    'Dataset': f'Dataset {1 if dataset=="AVO" else 2}',
                    'Method': method,
                    'Accuracy': row[acc_col],
                    'AUC': row[auc_col]
                })
    return pd.DataFrame(all_data)

# Create violin plots (only violin, no scatter)
def create_violin_plots():
    data = prepare_data()

    method_order = ['Single-dataset training', 'Combined training', 'AS-MMD (Proposed)']
    colors = ['#8dd3c7', '#fb8072', '#80b1d3']
    palette = dict(zip(method_order, colors))

    # y-limits
    acc_min, acc_max = data['Accuracy'].min(), data['Accuracy'].max()
    acc_range = acc_max - acc_min
    acc_ylim = [max(0, acc_min - 0.15 * acc_range), min(1, acc_max + 0.15 * acc_range)]

    auc_min, auc_max = data['AUC'].min(), data['AUC'].max()
    auc_range = auc_max - auc_min
    auc_ylim = [max(0, auc_min - 0.15 * auc_range), min(1, auc_max + 0.15 * auc_range)]

    # ---------------- Accuracy ----------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=data, x='Dataset', y='Accuracy', hue='Method',
        hue_order=method_order, palette=palette, ax=ax1,
        inner='box', cut=0, density_norm='width'   # cut=0
    )

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:3], labels[:3], title='Training Method',
               fontsize=11, title_fontsize=12, loc='lower right')

    ax1.set_title('Accuracy Distribution', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Dataset', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=16)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(acc_ylim)
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig('0930/violin_plot_accuracy.png', dpi=300, bbox_inches='tight')
    plt.savefig('0930/violin_plot_accuracy.pdf', bbox_inches='tight')
    plt.close()

    # ---------------- AUC ----------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=data, x='Dataset', y='AUC', hue='Method',
        hue_order=method_order, palette=palette, ax=ax2,
        inner='box', cut=0, density_norm='width'   # cut=0
    )

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:3], labels[:3], title='Training Method',
               fontsize=11, title_fontsize=12, loc='lower right')

    ax2.set_title('AUC Distribution', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Dataset', fontsize=16)
    ax2.set_ylabel('AUC', fontsize=16)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(auc_ylim)
    ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig('0930/violin_plot_auc.png', dpi=300, bbox_inches='tight')
    plt.savefig('0930/violin_plot_auc.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    create_violin_plots()
