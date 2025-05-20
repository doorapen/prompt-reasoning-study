import re
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def parse_summary_file(file_path, model_name_override=None):
    """Parses a summary.txt file to extract accuracies."""
    accuracies = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        dataset_name = None
        model_name = None
        
        for line in lines:
            if line.startswith("Model:"):
                match = re.search(r"Model: (.*?), Dataset: (.*)", line)
                if match:
                    model_name = match.group(1).strip()
                    dataset_name = match.group(2).strip()
                    break
        
        if not dataset_name:
            print(f"Could not parse dataset name from {file_path}")
            # Try to infer from path if possible, or skip this entry
            # For now, returning None if essential info is missing.
            return None
        
        # Use override if provided, otherwise use parsed model name
        accuracies['model'] = model_name_override if model_name_override else model_name
        if not accuracies['model']:
            print(f"Error: Model name could not be determined for {file_path}")
            return None
            
        accuracies['dataset'] = dataset_name
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            overall_match = re.match(r"Overall accuracy: (\d+\.\d+)", line)
            if overall_match:
                accuracies['overall'] = float(overall_match.group(1))
            
            prompt_acc_match = re.match(r"(.+?) accuracy: (\d+\.\d+)", line)
            if prompt_acc_match:
                prompt_type = prompt_acc_match.group(1).strip()
                acc_value = float(prompt_acc_match.group(2))
                if prompt_type.lower() != "overall":
                     accuracies[prompt_type] = acc_value

    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return None
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None
    
    if 'overall' not in accuracies and len(accuracies) <=2: # only model and dataset
        print(f"Warning: No accuracy data found in {file_path}. Check file content.")
        return None

    return accuracies

def main():
    results_base_dir = Path("results")
    models_to_plot = ["gpt-4", "deepseek-r1"]
    # Datasets to process from summary files (excluding MATH now)
    datasets_from_summary = ["gsm8k", "strategyqa", "commonsenseqa"] 
    all_datasets_for_plot_order = ["gsm8k", "math", "strategyqa", "commonsenseqa"] # For consistent plot order

    summary_file_map = {
        "gpt-4": {
            "gsm8k": "summary_20250512_124742.txt",
            # MATH will be hardcoded
            "strategyqa": "summary_20250518_212256.txt",
            "commonsenseqa": "summary_20250518_213403.txt" 
        },
        "deepseek-r1": {
            "gsm8k": "summary_20250512_125405.txt",
            # MATH will be hardcoded
            "strategyqa": "summary_20250518_212240.txt",
            "commonsenseqa": "summary_20250518_213623.txt" 
        }
    }
    
    all_data_records = []
    # Parse summary files for gsm8k, strategyqa, commonsenseqa
    for model_name in models_to_plot:
        for dataset_name in datasets_from_summary: # Iterate only over these datasets
            summary_filename = summary_file_map.get(model_name, {}).get(dataset_name)
            if not summary_filename:
                print(f"Warning: No summary file defined for {model_name} on {dataset_name}. Skipping.")
                continue
            
            summary_file_path = results_base_dir / dataset_name / model_name / summary_filename
            parsed_data = parse_summary_file(summary_file_path, model_name_override=model_name)
            if parsed_data:
                all_data_records.append(parsed_data)

    # Manually add MATH data based on user input
    math_data_input = { # Structure: {model: {prompt_type: accuracy, ...}, ...}
        "gpt-4": {
            "dataset": "math", "model": "gpt-4", "overall": 0.4415,
            "zero_shot": 0.5591, "few_shot": 0.5637, "cot": 0.3441, # Used 0.5637 for few_shot as per latest user input
            "self_consistency": 0.4000, # Added self_consistency
            "self_reflection": 0.3444, "react": 0.2796
        },
        "deepseek-r1": {
            "dataset": "math", "model": "deepseek-r1", "overall": 0.5018,
            "zero_shot": 0.6122, "few_shot": 0.6136, "cot": 0.4211,
            "self_consistency": 0.4000, "self_reflection": 0.4286, "react": 0.5366
        }
    }
    for model_name in models_to_plot:
        if model_name in math_data_input:
            all_data_records.append(math_data_input[model_name])

    print("Collected data for plotting (including manual MATH data):")
    for item in all_data_records:
        print(item)

    if not all_data_records:
        print("No data collected. Exiting.")
        return

    plot_df_list = []
    for record in all_data_records:
        dataset = record['dataset']
        model = record['model']
        for key, value in record.items():
            if key not in ['dataset', 'model', 'overall']:
                plot_df_list.append({
                    'dataset': dataset, 
                    'model': model,
                    'prompt_type': key, 
                    'accuracy': value
                })
    
    if not plot_df_list:
        print("No prompt-specific accuracy data found to plot.")
        overall_plot_df_list = []
        for record in all_data_records:
            if 'overall' in record:
                overall_plot_df_list.append({
                    'dataset': record['dataset'], 
                    'model': record['model'], 
                    'prompt_type': 'overall', 
                    'accuracy': record['overall']
                })
        if overall_plot_df_list:
            print("Plotting overall accuracies instead.")
            plot_df = pd.DataFrame(overall_plot_df_list)
        else:
            print("No data to plot at all.")
            return
    else:
        plot_df = pd.DataFrame(plot_df_list)

    print("\nPlotting DataFrame:")
    print(plot_df)

    if plot_df.empty:
        print("Plotting DataFrame is empty. Cannot generate plot.")
        return

    output_plot_dir = Path("plots")
    output_plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(18, 10))
    
    if not plot_df.empty:
        g = sns.catplot(x="dataset", y="accuracy", hue="prompt_type", col="model", 
                        data=plot_df, kind="bar", height=6, aspect=1.2, 
                        palette="Spectral",
                        legend_out=True,
                        order=all_datasets_for_plot_order) # Use all_datasets_for_plot_order for x-axis order
        g.set_axis_labels("Dataset", "Accuracy")
        g.set_titles("Model: {col_name}")
        g.fig.suptitle('Model Accuracy Comparison: Dataset vs. Prompt Type', y=1.03, fontsize=16)
        g.add_legend(title="Prompt Type")
        plot_filename = output_plot_dir / "accuracy_comparison_all_models_all_datasets.png" # New filename
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"\nPlot saved to {plot_filename}")
        plt.show()
    else:
        print("No data available for plotting after processing.")

if __name__ == "__main__":
    main() 