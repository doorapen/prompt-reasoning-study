"""
Visualization utilities for experiment results.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from collections import Counter


def ensure_plot_dir():
    """Create plots directory if it doesn't exist."""
    os.makedirs("plots", exist_ok=True)


def plot_accuracy_comparison(
    results_by_prompt: Dict[str, Dict[str, Any]],
    model_name: str,
    dataset_name: str
):
    """
    Plot accuracy comparison across prompt types.
    
    Args:
        results_by_prompt: Dictionary mapping prompt types to their metrics
        model_name: Name of the model
        dataset_name: Name of the dataset
    """
    ensure_plot_dir()
    
    # Create a DataFrame for plotting
    data = []
    for prompt_type, metrics in results_by_prompt.items():
        data.append({
            "Prompt Type": prompt_type,
            "Accuracy": metrics["accuracy"],
            "Total": metrics["total"]
        })
    
    df = pd.DataFrame(data)
    
    # Sort by accuracy
    df = df.sort_values("Accuracy", ascending=False)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    bar_plot = sns.barplot(
        x="Prompt Type", 
        y="Accuracy", 
        hue="Prompt Type",
        data=df,
        palette="viridis",
        legend=False
    )
    
    # Add value labels on top of bars
    for i, p in enumerate(bar_plot.patches):
        bar_plot.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=10
        )
    
    # Formatting
    plt.title(f"Accuracy by Prompt Type ({model_name} on {dataset_name})")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"plots/{model_name}_{dataset_name}_accuracy_comparison.png", dpi=300)
    plt.close()


def plot_error_distribution(
    results: List[Dict[str, Any]],
    model_name: str,
    dataset_name: str
):
    """
    Plot distribution of error types across prompt families.
    
    Args:
        results: List of result dictionaries
        model_name: Name of the model
        dataset_name: Name of the dataset
    """
    ensure_plot_dir()
    
    # Group results by prompt type and count error types
    error_counts = {}
    
    for result in results:
        prompt_type = result.get("prompt_type")
        if not prompt_type:
            continue
            
        # Check if the result has an error_type
        error_type = result.get("error_type")
        
        # If result is correct, error_type is None
        if result.get("answer") == result.get("true_answer"):
            error_type = "correct"
        elif not error_type:
            error_type = "UNKNOWN"
            
        if prompt_type not in error_counts:
            error_counts[prompt_type] = Counter()
            
        error_counts[prompt_type][error_type] += 1
    
    # Convert to DataFrame for plotting
    data = []
    for prompt_type, counts in error_counts.items():
        total = sum(counts.values())
        for error_type, count in counts.items():
            data.append({
                "Prompt Type": prompt_type,
                "Error Type": error_type,
                "Count": count,
                "Percentage": count / total * 100 if total > 0 else 0
            })
    
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot percentages as stacked bars
    if len(df) == 0:
        print("No error data to plot")
        return
        
    pivot_df = df.pivot_table(
        index="Prompt Type", 
        columns="Error Type", 
        values="Percentage",
        fill_value=0
    )
    
    # Check if 'correct' is in columns before reordering
    if 'correct' in pivot_df.columns:
        # Ensure correct is on top (bottom in stacked bar)
        cols = [c for c in pivot_df.columns if c != "correct"] + ["correct"]
        pivot_df = pivot_df[cols]
    
    ax = pivot_df.plot(
        kind="bar", 
        stacked=True,
        figsize=(12, 8),
        colormap="viridis"
    )
    
    # Formatting
    plt.title(f"Error Distribution by Prompt Type ({model_name} on {dataset_name})")
    plt.xlabel("Prompt Type")
    plt.ylabel("Percentage (%)")
    plt.legend(title="Error Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"plots/{model_name}_{dataset_name}_error_distribution.png", dpi=300)
    plt.close()


def plot_rationale_length(
    results: List[Dict[str, Any]],
    model_name: str,
    dataset_name: str
):
    """
    Plot rationale length vs. correctness for CoT variants.
    
    Args:
        results: List of result dictionaries (must include reasoning_length)
        model_name: Name of the model
        dataset_name: Name of the dataset
    """
    ensure_plot_dir()
    
    # Filter results with reasoning length
    filtered_results = [
        r for r in results 
        if "reasoning_length" in r
    ]
    
    if not filtered_results:
        print("No results with reasoning_length found for plotting")
        return
    
    # Create a DataFrame for plotting
    data = []
    for result in filtered_results:
        prompt_type = result.get("prompt_type")
        is_correct = result.get("answer") == result.get("true_answer")
        
        data.append({
            "Prompt Type": prompt_type,
            "Reasoning Length": result["reasoning_length"],
            "Correct": "Yes" if is_correct else "No"
        })
    
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Check if we have both correct and incorrect examples
    has_both_outcomes = "Yes" in df["Correct"].values and "No" in df["Correct"].values
    
    # Fix: Only use hue if we have both correct and incorrect examples
    if has_both_outcomes:
        sns.boxplot(
            x="Prompt Type",
            y="Reasoning Length",
            hue="Correct",
            data=df,
            palette=["#FF9999", "#66B2FF"]
        )
    else:
        # If we only have one outcome, don't use hue
        sns.boxplot(
            x="Prompt Type",
            y="Reasoning Length",
            data=df
        )
    
    # Formatting
    plt.title(f"Reasoning Length by Prompt Type and Correctness ({model_name} on {dataset_name})")
    plt.xlabel("Prompt Type")
    plt.ylabel("Reasoning Length (tokens)")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"plots/{model_name}_{dataset_name}_rationale_length.png", dpi=300)
    plt.close()
    
    # Additional plot: Scatter plot with trend line (linear instead of logistic to avoid statsmodels dependency)
    plt.figure(figsize=(10, 6))
    
    for prompt_type in df["Prompt Type"].unique():
        prompt_df = df[df["Prompt Type"] == prompt_type]
        
        # Use linear regression instead of logistic to avoid statsmodels dependency
        sns.regplot(
            x="Reasoning Length",
            y=[1 if c == "Yes" else 0 for c in prompt_df["Correct"]],
            data=prompt_df,
            label=prompt_type,
            scatter=True,
            scatter_kws={"alpha": 0.3},
            logistic=False  # Changed to False to avoid statsmodels dependency
        )
    
    plt.title(f"Relationship Between Reasoning Length and Correctness ({model_name} on {dataset_name})")
    plt.xlabel("Reasoning Length (tokens)")
    plt.ylabel("Probability of Correctness")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"plots/{model_name}_{dataset_name}_length_vs_correctness.png", dpi=300)
    plt.close()


def plot_consistency_metrics(
    results: List[Dict[str, Any]],
    model_name: str,
    dataset_name: str
):
    """
    Plot consistency metrics for self-consistency results.
    
    Args:
        results: List of result dictionaries (must be from self-consistency)
        model_name: Name of the model
        dataset_name: Name of the dataset
    """
    ensure_plot_dir()
    
    # Filter only self-consistency results
    sc_results = [r for r in results if r.get("prompt_type") == "self_consistency"]
    
    if not sc_results:
        print("No self-consistency results found for plotting")
        return
    
    # Compute agreement rates
    agreement_rates = []
    is_correct = []
    
    for result in sc_results:
        if "sample_outputs" not in result:
            continue
            
        # Count answers in samples
        answers = [r.get("answer", "") for r in result.get("sample_outputs", [])]
        counter = Counter(answers)
        
        # Calculate agreement rate
        most_common = counter.most_common(1)[0]
        agreement_rate = most_common[1] / len(answers)
        
        agreement_rates.append(agreement_rate)
        is_correct.append(result.get("answer") == result.get("true_answer"))
    
    # Create DataFrame
    df = pd.DataFrame({
        "Agreement Rate": agreement_rates,
        "Correct": ["Yes" if c else "No" for c in is_correct]
    })
    
    # Plot histogram of agreement rates by correctness
    plt.figure(figsize=(10, 6))
    
    sns.histplot(
        data=df,
        x="Agreement Rate",
        hue="Correct",
        multiple="stack",
        bins=10,
        palette=["#FF9999", "#66B2FF"]
    )
    
    plt.title(f"Distribution of Agreement Rates ({model_name} on {dataset_name})")
    plt.xlabel("Agreement Rate")
    plt.ylabel("Count")
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"plots/{model_name}_{dataset_name}_agreement_rates.png", dpi=300)
    plt.close()


def accuracy_bar(df, outfile):
    """Legacy function for compatibility."""
    import seaborn as sns, matplotlib.pyplot as plt
    sns.barplot(data=df, x="family", y="correct", hue="model", errorbar="ci")
    plt.ylabel("Exact-Match Accuracy")
    plt.savefig(outfile, bbox_inches="tight", dpi=300) 