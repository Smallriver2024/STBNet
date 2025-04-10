import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import warnings
import itertools
warnings.filterwarnings('ignore')

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def main():
    start_time = datetime.datetime.now()
    print(f"SHAP Visualization Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output directory specifically for dependence plots
    dependence_dir = f"{output_dir}/dependence_plots"
    os.makedirs(dependence_dir, exist_ok=True)
    
    # Create PDF file
    pdf_path = f"{output_dir}/shap_visualizations.pdf"
    pdf = PdfPages(pdf_path)
    
    # Create separate PDF file for all dependence plots
    dependence_pdf_path = f"{output_dir}/shap_dependence_plots.pdf"
    dependence_pdf = PdfPages(dependence_pdf_path)
    
    # 1. Load training data
    try:
        train_data = pd.read_excel("train.xlsx")
        if 'Group' in train_data.columns:
            y_train = train_data['Group']
            x_train = train_data.drop(['Group'], axis=1)
        else:
            x_train = train_data
            y_train = None
            
        feature_names = list(x_train.columns)
        print(f"Dataset: {len(x_train)} samples, {len(feature_names)} features")
        print(f"Feature list: {', '.join(feature_names)}")
        print()
    except Exception as e:
        print(f"Error loading training data: {e}")
        return
    
    # 2. Load SHAP values from .npy file
    print(f"Loading SHAP values from shap_values_array.npy file")
    try:
        # Directly load the numpy array file
        shap_values_array = np.load("shap_values_array.npy")
        
        print(f"SHAP values shape: {shap_values_array.shape}")
        
        # Check if dimensions match
        if shap_values_array.shape[1] != len(feature_names):
            print(f"Warning: SHAP values dimensions ({shap_values_array.shape[1]}) don't match feature count ({len(feature_names)})")
            
            # Adjust if needed
            if shap_values_array.shape[1] < len(feature_names):
                print(f"Truncating feature list to match SHAP values")
                feature_names = feature_names[:shap_values_array.shape[1]]
            else:
                print(f"Truncating SHAP values to match feature list")
                shap_values_array = shap_values_array[:, :len(feature_names)]
            
            print(f"Adjusted dimensions - SHAP values: {shap_values_array.shape}, Features: {len(feature_names)}")
        
        # Use mean prediction as base value since we don't have it stored
        base_value = np.mean(shap_values_array)
        print(f"Using {base_value:.4f} as base value (mean of SHAP values)")
        
    except Exception as e:
        print(f"Error loading SHAP values from NPY file: {e}")
        return
    
    # 3. Generate visualizations
    plt.rcParams.update({'font.size': 12})
    
    # 3.1 Summary plot (beeswarm style)
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_array, x_train, feature_names=feature_names, show=False)
        plt.title("SHAP Summary Plot", fontsize=14)
        plt.tight_layout()
        output_path = f"{output_dir}/shap_summary_plot.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        pdf.savefig()
        plt.close()
        print(f"✓ Created SHAP summary plot: {output_path}")
    except Exception as e:
        print(f"Creating SHAP summary plot failed: {str(e)}")
    
    # 3.2 Summary plot (bar style)
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_array, x_train, feature_names=feature_names, 
                        plot_type="bar", show=False)
        plt.title("SHAP Feature Importance", fontsize=14)
        plt.tight_layout()
        output_path = f"{output_dir}/shap_bar_plot.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        pdf.savefig()
        plt.close()
        print(f"✓ Created SHAP bar plot: {output_path}")
    except Exception as e:
        print(f"Creating SHAP bar summary failed: {str(e)}")
    
    # 3.3 Calculate feature importance
    feature_importance = np.mean(np.abs(shap_values_array), axis=0)
    
    # Get indices of features sorted by importance
    sorted_indices = np.argsort(-feature_importance)
    
    # 3.4 Dependence plots for ALL features
    print("\nGenerating dependence plots for all features...")
    
    # Generate all possible feature pairs for interaction plots
    all_pairs = list(itertools.permutations(range(len(feature_names)), 2))
    
    # Create individual dependence plots for each feature
    for i, feature_idx in enumerate(range(len(feature_names))):
        feature_name = feature_names[feature_idx]
        try:
            plt.figure(figsize=(10, 8))
            # Create dependence plot without interaction
            shap.dependence_plot(
                ind=feature_idx,
                shap_values=shap_values_array,
                features=x_train,
                feature_names=feature_names,
                interaction_index=None,
                show=False
            )
            plt.title(f"SHAP Dependence Plot: {feature_name}", fontsize=14)
            plt.tight_layout()
            output_path = f"{dependence_dir}/dependence_{feature_name}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            dependence_pdf.savefig()
            # Add to main PDF only for top 5 features
            if feature_idx in sorted_indices[:5]:
                pdf.savefig()
            plt.close()
            print(f"✓ Created dependence plot for feature '{feature_name}': {output_path}")
        except Exception as e:
            print(f"Creating SHAP dependence plot for feature '{feature_name}' failed: {str(e)}")
    
    # 3.5 Create interaction dependence plots for top combinations
    print("\nGenerating interaction dependence plots...")
    
    # Take the top 10 feature pairs (or all if fewer)
    top_pairs = []
    pair_count = 0
    
    # First prioritize pairs that include at least one top feature
    top_features = set(sorted_indices[:3])  # Top 3 features
    for idx1, idx2 in all_pairs:
        if idx1 in top_features or idx2 in top_features:
            top_pairs.append((idx1, idx2))
            pair_count += 1
            if pair_count >= 15:  # Limit to 15 interaction plots
                break
    
    # Generate interaction plots
    for feature_idx, interaction_idx in top_pairs:
        feature_name = feature_names[feature_idx]
        interaction_name = feature_names[interaction_idx]
        
        try:
            plt.figure(figsize=(10, 8))
            # Create dependence plot with interaction
            shap.dependence_plot(
                ind=feature_idx,
                shap_values=shap_values_array,
                features=x_train,
                feature_names=feature_names,
                interaction_index=interaction_idx,
                show=False
            )
            plt.title(f"SHAP Dependence Plot: {feature_name} (with {interaction_name} interaction)", fontsize=14)
            plt.tight_layout()
            output_path = f"{dependence_dir}/dependence_{feature_name}_with_{interaction_name}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            dependence_pdf.savefig()
            plt.close()
            print(f"✓ Created interaction dependence plot: {feature_name} with {interaction_name}: {output_path}")
        except Exception as e:
            print(f"Creating interaction dependence plot for '{feature_name}' with '{interaction_name}' failed: {str(e)}")
    
    # 3.6 Force plots for individual samples
    sample_indices = [0, len(x_train)//2, len(x_train)-1]  # First, middle, last
    for idx in sample_indices:
        try:
            plt.figure(figsize=(12, 4))
            # Create force plot with matplotlib rendering
            shap.force_plot(
                base_value=base_value,
                shap_values=shap_values_array[idx],
                features=x_train.iloc[idx],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f"SHAP Force Plot: Sample #{idx}", fontsize=14)
            plt.tight_layout()
            output_path = f"{output_dir}/shap_force_plot_sample_{idx}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            pdf.savefig()
            plt.close()
            print(f"✓ Created force plot for sample {idx}: {output_path}")
        except Exception as e:
            print(f"Error creating force plot for sample {idx}: {str(e)}")
            
            # Try alternative method for force plot
            try:
                plt.figure(figsize=(12, 4))
                
                # Manual force plot creation
                feature_values = x_train.iloc[idx].values
                shap_values_sample = shap_values_array[idx]
                
                # Sort by absolute value
                indices = np.argsort(np.abs(shap_values_sample))[::-1]
                sorted_features = [feature_names[i] for i in indices]
                sorted_values = [shap_values_sample[i] for i in indices]
                sorted_data = [feature_values[i] for i in indices]
                
                # Create bar plot
                colors = ['red' if val < 0 else 'blue' for val in sorted_values]
                plt.barh(range(len(sorted_features)), sorted_values, color=colors)
                plt.yticks(range(len(sorted_features)), sorted_features)
                plt.title(f"SHAP Values for Sample #{idx} (Alternative Plot)", fontsize=14)
                plt.xlabel("SHAP Value", fontsize=12)
                
                # Add feature values as text
                for i, val in enumerate(sorted_data):
                    plt.text(0, i, f" = {val:.2f}", va='center')
                
                plt.tight_layout()
                output_path = f"{output_dir}/shap_force_plot_alt_sample_{idx}.pdf"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                pdf.savefig()
                plt.close()
                print(f"✓ Created alternative force plot for sample {idx}: {output_path}")
            except Exception as e2:
                print(f"Error creating alternative force plot for sample {idx}: {str(e2)}")
    
    # 3.7 Export SHAP values to CSV
    try:
        shap_df = pd.DataFrame(shap_values_array, columns=feature_names)
        csv_path = f"{output_dir}/shap_values.csv"
        shap_df.to_csv(csv_path, index=False)
        print(f"✓ SHAP values exported to: {csv_path}")
        
        # Export feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        importance_path = f"{output_dir}/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"✓ Feature importance exported to: {importance_path}")
    except Exception as e:
        print(f"Error exporting data to CSV: {str(e)}")
    
    # Close PDF files
    pdf.close()
    dependence_pdf.close()
    
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nSHAP visualization completed in {duration:.2f} seconds")
    print(f"Main PDF report saved to: {pdf_path}")
    print(f"Dependence plots PDF saved to: {dependence_pdf_path}")
    print(f"Individual plots saved to: {dependence_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()