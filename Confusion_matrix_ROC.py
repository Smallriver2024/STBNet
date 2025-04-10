import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Create a subdirectory for saving PDFs
output_dir = "confusion_matrix_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Read the previously generated CSV files
# Note: Change these to your actual file paths
train_results_path = "all_train_results.csv"
val_results_path = "all_val_results.csv"

train_results = pd.read_csv(train_results_path)
val_results = pd.read_csv(val_results_path)

# Get all model names
model_names = [col.replace('_proba', '') for col in train_results.columns if col.endswith('_proba')]

# Timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_confusion_matrix(y_true, y_pred, model_name, dataset_type, normalize=False):
    """
    Plot confusion matrix
    
    Parameters:
    y_true: True labels
    y_pred: Predicted labels
    model_name: Name of the model
    dataset_type: Type of dataset (Training or Validation)
    normalize: Whether to normalize the matrix
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate row-wise percentages (percentage within each true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_percentages = cm / row_sums * 100
    
    # Create annotations
    if normalize:
        # For normalized matrices, show normalized value and row percentage
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        annot = np.empty_like(cm, dtype=object)
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                annot[i, j] = f'{cm_norm[i, j]:.2f}\n({row_percentages[i, j]:.1f}%)'
        title = f'{model_name} - {dataset_type} Normalized Confusion Matrix'
    else:
        # For regular matrices, show count and row percentage
        annot = np.empty_like(cm, dtype=object)
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                annot[i, j] = f'{cm[i, j]}\n({row_percentages[i, j]:.1f}%)'
        title = f'{model_name} - {dataset_type} Confusion Matrix'
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot heatmap with custom annotations
    ax = sns.heatmap(cm if not normalize else cm_norm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['PSI', 'STB'],
                yticklabels=['PSI', 'STB'], annot_kws={"size": 20, "weight": "bold"})
    
    # Set title and labels
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    
    return plt.gcf()

# Create a function to apply threshold to probabilities to get predicted labels
def apply_threshold(y_proba, threshold):
    return (y_proba >= threshold).astype(int)

# Find optimal threshold using Youden's Index
def find_optimal_threshold(y_true, y_proba):
    """
    Find optimal threshold using Youden's Index
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    return thresholds[optimal_idx]

# Calculate optimal threshold for each model
thresholds = {}
for model_name in model_names:
    y_train_true = train_results['y_train']
    y_train_proba = train_results[f'{model_name}_proba']
    threshold = find_optimal_threshold(y_train_true, y_train_proba)
    thresholds[model_name] = threshold
    print(f"Optimal threshold for {model_name}: {threshold:.4f}")

# Set publication-quality figure parameters
plt.rcParams.update({
    'font.family': 'Arial',        # 统一字体类型
    'font.size': 12,               # 基础字号（影响图例、刻度等）
    'axes.titlesize': 18,          # 标题字号
    'axes.labelsize': 16,          # 坐标轴标签字号
    'xtick.labelsize': 18,         # X轴刻度
    'ytick.labelsize': 18,         # Y轴刻度
    'legend.fontsize': 12,         # 图例字号
    'figure.dpi': 600,             # 输出分辨率
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',       # 自动裁剪白边
    'axes.linewidth': 1.5          # 坐标轴线宽
})

# Create a PDF for each model
for model_name in model_names:
    pdf_path = os.path.join(output_dir, f"{model_name}_confusion_matrices_{timestamp}.pdf")
    
    with PdfPages(pdf_path) as pdf:
        # Training set
        y_train_true = train_results['y_train']
        y_train_proba = train_results[f'{model_name}_proba']
        y_train_pred = apply_threshold(y_train_proba, thresholds[model_name])
        
        # Validation set
        y_val_true = val_results['y_val']
        y_val_proba = val_results[f'{model_name}_proba']
        y_val_pred = apply_threshold(y_val_proba, thresholds[model_name])
        
        # Plot and save non-normalized confusion matrices
        train_cm_fig = plot_confusion_matrix(y_train_true, y_train_pred, model_name, 'Training Set', normalize=False)
        pdf.savefig(train_cm_fig)
        plt.close()
        
        val_cm_fig = plot_confusion_matrix(y_val_true, y_val_pred, model_name, 'Validation Set', normalize=False)
        pdf.savefig(val_cm_fig)
        plt.close()
        
        # Plot and save normalized confusion matrices
        train_cm_norm_fig = plot_confusion_matrix(y_train_true, y_train_pred, model_name, 'Training Set', normalize=True)
        pdf.savefig(train_cm_norm_fig)
        plt.close()
        
        val_cm_norm_fig = plot_confusion_matrix(y_val_true, y_val_pred, model_name, 'Validation Set', normalize=True)
        pdf.savefig(val_cm_norm_fig)
        plt.close()
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = f'{model_name} Confusion Matrices'
        d['Author'] = 'Automated Script'
        d['Subject'] = 'Model Evaluation Results'
        d['Keywords'] = 'confusion matrix, classification, machine learning'
        d['CreationDate'] = datetime.now()
        d['ModDate'] = datetime.now()
    
    print(f"Created confusion matrix PDF for model {model_name}: {pdf_path}")

# Create a single PDF with confusion matrices for all models (for comparison)
all_models_pdf_path = os.path.join(output_dir, f"all_models_confusion_matrices_{timestamp}.pdf")

with PdfPages(all_models_pdf_path) as pdf:
    # Create comparison plots for training and validation sets
    for dataset_type, results, y_true_col in [
        ('Training Set', train_results, 'y_train'),
        ('Validation Set', val_results, 'y_val')
    ]:
        # Non-normalized confusion matrices
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Confusion Matrices for All Models on {dataset_type}", fontsize=16, fontweight='bold')
        
        for i, model_name in enumerate(model_names):
            plt.subplot(2, (len(model_names) + 1) // 2, i + 1)
            
            y_true = results[y_true_col]
            y_proba = results[f'{model_name}_proba']
            y_pred = apply_threshold(y_proba, thresholds[model_name])
            
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate row-wise percentages (percentage within each true class)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_percentages = cm / row_sums * 100
            
            # Create annotations with counts and percentages
            annot = np.empty_like(cm, dtype=object)
            for i_cm in range(len(cm)):
                for j_cm in range(len(cm[i_cm])):
                    annot[i_cm, j_cm] = f'{cm[i_cm, j_cm]}\n({row_percentages[i_cm, j_cm]:.1f}%)'
            
            sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                        xticklabels=['PSI', 'STB'],
                        yticklabels=['PSI', 'STB'],
                        annot_kws={"size": 20, "weight": "bold"})
            
            plt.title(f'{model_name} (Threshold: {thresholds[model_name]:.3f})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
        
        # Normalized confusion matrices
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Normalized Confusion Matrices for All Models on {dataset_type}", fontsize=16, fontweight='bold')
        
        for i, model_name in enumerate(model_names):
            plt.subplot(2, (len(model_names) + 1) // 2, i + 1)
            
            y_true = results[y_true_col]
            y_proba = results[f'{model_name}_proba']
            y_pred = apply_threshold(y_proba, thresholds[model_name])
            
            cm = confusion_matrix(y_true, y_pred)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Calculate row-wise percentages (percentage within each true class)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_percentages = cm / row_sums * 100
            
            # Create annotations with normalized values and row percentages
            annot = np.empty_like(cm, dtype=object)
            for i_cm in range(len(cm)):
                for j_cm in range(len(cm[i_cm])):
                    annot[i_cm, j_cm] = f'{cm_norm[i_cm, j_cm]:.2f}\n({row_percentages[i_cm, j_cm]:.1f}%)'
            
            sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                        xticklabels=['PSI', 'STB'],
                        yticklabels=['PSI', 'STB'],
                        annot_kws={"size": 20, "weight": "bold"})
            
            plt.title(f'{model_name} (Threshold: {thresholds[model_name]:.3f})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
    
    # Set PDF metadata
    d = pdf.infodict()
    d['Title'] = 'Comparison of Confusion Matrices for All Models'
    d['Author'] = 'Automated Script'
    d['Subject'] = 'Model Evaluation Results'
    d['Keywords'] = 'confusion matrix, classification, machine learning, comparison'
    d['CreationDate'] = datetime.now()
    d['ModDate'] = datetime.now()

print(f"Created comparison PDF for all models' confusion matrices: {all_models_pdf_path}")

# Additional feature: Plot ROC curves and save them in the same folder
from sklearn.metrics import roc_curve, auc

roc_pdf_path = os.path.join(output_dir, f"roc_curves_{timestamp}.pdf")

with PdfPages(roc_pdf_path) as pdf:
    # Plot ROC curves for training and validation sets
    for dataset_type, results, y_true_col in [
        ('Training Set', train_results, 'y_train'),
        ('Validation Set', val_results, 'y_val')
    ]:
        plt.figure(figsize=(6, 6))
        
        for model_name in model_names:
            y_true = results[y_true_col]
            y_proba = results[f'{model_name}_proba']
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line (random guess)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC Curves - {dataset_type}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    # Set PDF metadata
    d = pdf.infodict()
    d['Title'] = 'ROC Curves'
    d['Author'] = 'Automated Script'
    d['Subject'] = 'Model Evaluation Results'
    d['Keywords'] = 'ROC, AUC, classification, machine learning'
    d['CreationDate'] = datetime.now()
    d['ModDate'] = datetime.now()

# Generate individual high-quality PNG files for each plot (useful for publications)
for model_name in model_names:
    for dataset_type, results, y_true_col in [
        ('Training_Set', train_results, 'y_train'),
        ('Validation_Set', val_results, 'y_val')
    ]:
        y_true = results[y_true_col]
        y_proba = results[f'{model_name}_proba']
        y_pred = apply_threshold(y_proba, thresholds[model_name])
        
        # Non-normalized confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate row-wise percentages (percentage within each true class)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_percentages = cm / row_sums * 100
        
        # Create annotations with counts and row percentages
        annot = np.empty_like(cm, dtype=object)
        for i_cm in range(len(cm)):
            for j_cm in range(len(cm[i_cm])):
                annot[i_cm, j_cm] = f'{cm[i_cm, j_cm]}\n({row_percentages[i_cm, j_cm]:.1f}%)'
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                    xticklabels=['PSI', 'STB'],
                    yticklabels=['PSI', 'STB'],
                    annot_kws={"size": 20, "weight": "bold"})
        plt.title(f'{model_name} - {dataset_type.replace("_", " ")} Confusion Matrix', fontsize=18, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=16)
        plt.ylabel('True Label', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_{dataset_type}_confusion_matrix.pdf"), dpi=600)
        plt.close()
        
        # Normalized confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Calculate row-wise percentages (percentage within each true class)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_percentages = cm / row_sums * 100
        
        # Create annotations with normalized values and row percentages
        annot = np.empty_like(cm, dtype=object)
        for i_cm in range(len(cm)):
            for j_cm in range(len(cm[i_cm])):
                annot[i_cm, j_cm] = f'{cm_norm[i_cm, j_cm]:.2f}\n({row_percentages[i_cm, j_cm]:.1f}%)'
        
        sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                    xticklabels=['PSI', 'STB'],
                    yticklabels=['PSI', 'STB'],
                    annot_kws={"size": 20, "weight": "bold"})
        plt.title(f'{model_name} - {dataset_type.replace("_", " ")} Normalized Confusion Matrix', fontsize=18, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=16)
        plt.ylabel('True Label', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_{dataset_type}_normalized_confusion_matrix.pdf"), dpi=600)
        plt.close()

# Save ROC curves as individual PDF files
for dataset_type, results, y_true_col in [
    ('Training_Set', train_results, 'y_train'),
    ('Validation_Set', val_results, 'y_val')
]:
    plt.figure(figsize=(6, 6))
    
    for model_name in model_names:
        y_true = results[y_true_col]
        y_proba = results[f'{model_name}_proba']
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line (random guess)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f'ROC Curves - {dataset_type.replace("_", " ")}', fontsize=18, fontweight='bold')
    plt.legend(loc="lower right")
    #plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ROC_curves_{dataset_type}.pdf"), dpi=600)
    plt.close()

print(f"All graphical files have been saved in the '{output_dir}' folder.")
print("The script has generated:")
print(f"- Individual PDF files for each model's confusion matrices")
print(f"- A comparison PDF with all models' confusion matrices")
print(f"- A PDF with ROC curves for all models")
print(f"- High-quality PNG files for each plot (suitable for publication)")