"""
Main script to run all experiments.
"""
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd

from src.prompts import get_prompt_template
from src.inference import get_model_interface, run_inference
from src.eval import evaluate_gsm8k, evaluate_strategyqa, bootstrap_ci, mcnemar_test
from src.plots import plot_accuracy_comparison, plot_error_distribution, plot_rationale_length


def load_dataset(dataset_name: str, split: str = "test", max_examples: int = None) -> List[Dict[str, Any]]:
    """
    Load a dataset for evaluation.
    
    Args:
        dataset_name: Name of the dataset (gsm8k, strategyqa)
        split: Data split to use (train, test, dev)
        max_examples: Maximum number of examples to load (for testing)
        
    Returns:
        List of examples
    """
    # Path to datasets
    data_dir = os.path.join("data", dataset_name)
    os.makedirs(data_dir, exist_ok=True)
    
    # Load the dataset based on name
    if dataset_name == "gsm8k":
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main")
        examples = dataset[split]
        
        # Convert to list of dictionaries
        data = []
        for i, ex in enumerate(examples):
            if max_examples and i >= max_examples:
                break
            data.append({
                "id": f"gsm8k_{split}_{i}",
                "question": ex["question"],
                "answer": ex["answer"].split("####")[-1].strip()
            })
        
    elif dataset_name == "strategyqa":
        from datasets import load_dataset
        dataset = load_dataset("stanford-crfm/strategyqa")
        examples = dataset[split]
        
        # Convert to list of dictionaries
        data = []
        for i, ex in enumerate(examples):
            if max_examples and i >= max_examples:
                break
            data.append({
                "id": f"strategyqa_{split}_{i}",
                "question": ex["question"],
                "answer": "yes" if ex["answer"] else "no"
            })
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return data


def save_results(results: List[Dict[str, Any]], 
                model_name: str, 
                dataset_name: str, 
                prompt_type: str):
    """Save results to a JSON file."""
    results_dir = os.path.join("results", dataset_name, model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prompt_type}_{timestamp}.json"
    
    with open(os.path.join(results_dir, filename), "w") as f:
        json.dump(results, f, indent=2)


def save_batch(batch: List[Dict[str, Any]], 
              model_name: str, 
              dataset_name: str, 
              prompt_type: str):
    """Save a batch of results (used as a callback during inference)."""
    results_dir = os.path.join("results", dataset_name, model_name, prompt_type)
    os.makedirs(results_dir, exist_ok=True)
    
    # Use first item's ID for the batch file
    batch_id = batch[0]["id"]
    filename = f"batch_{batch_id}.json"
    
    with open(os.path.join(results_dir, filename), "w") as f:
        json.dump(batch, f, indent=2)


def run_experiment(
    model_name: str,
    dataset_name: str,
    prompt_types: List[str],
    max_examples: int = None,
    save_intermediate: bool = True
):
    """
    Run a complete experiment for a model on a dataset with multiple prompt types.
    
    Args:
        model_name: Name of the model (gpt-4, deepseek-r1)
        dataset_name: Name of the dataset (gsm8k, strategyqa)
        prompt_types: List of prompt types to evaluate
        max_examples: Maximum number of examples to process
        save_intermediate: Whether to save intermediate results
    """
    # Load dataset
    dataset = load_dataset(dataset_name, max_examples=max_examples)
    print(f"Loaded {len(dataset)} examples from {dataset_name}")
    
    # Load few-shot examples if needed (for few-shot prompting)
    if "few_shot" in prompt_types:
        few_shot_examples = load_dataset(dataset_name, split="train", max_examples=10)
        print(f"Loaded {len(few_shot_examples)} few-shot examples")
    else:
        few_shot_examples = None
    
    # Get model interface
    model = get_model_interface(model_name)
    
    all_results = []
    
    # Run inference for each prompt type
    for prompt_type in prompt_types:
        print(f"Running {model_name} on {dataset_name} with {prompt_type} prompting")
        
        # Get prompt template
        prompt_template = get_prompt_template(prompt_type)
        
        # Define batch callback if needed
        batch_callback = None
        if save_intermediate:
            batch_callback = lambda batch: save_batch(
                batch, model_name, dataset_name, prompt_type
            )
        
        # Run inference
        results = run_inference(
            model=model,
            prompt_template=prompt_template,
            dataset=dataset,
            examples=few_shot_examples if prompt_type == "few_shot" else None,
            batch_callback=batch_callback,
            batch_size=5  # Save every 5 examples
        )
        
        # Save results
        save_results(results, model_name, dataset_name, prompt_type)
        all_results.extend(results)
    
    # Run evaluation based on dataset
    if dataset_name == "gsm8k":
        metrics = evaluate_gsm8k(all_results)
    elif dataset_name == "strategyqa":
        metrics = evaluate_strategyqa(all_results)
    
    # Generate evaluation report
    print("\n=== Evaluation Results ===")
    print(f"Model: {model_name}, Dataset: {dataset_name}")
    print(f"Total examples: {metrics['total']}")
    print(f"Overall accuracy: {metrics['accuracy']:.4f}")
    
    # Compare prompt types
    results_by_prompt = {}
    for prompt_type in prompt_types:
        prompt_results = [r for r in all_results if r.get("prompt_type") == prompt_type]
        if dataset_name == "gsm8k":
            prompt_metrics = evaluate_gsm8k(prompt_results)
        else:
            prompt_metrics = evaluate_strategyqa(prompt_results)
        
        results_by_prompt[prompt_type] = prompt_metrics
        print(f"\n{prompt_type} accuracy: {prompt_metrics['accuracy']:.4f}")
        
        # Get confidence intervals
        ci = bootstrap_ci(
            prompt_results,
            lambda res: sum(1 for r in res if r.get("answer") == r.get("true_answer")) / len(res)
        )
        print(f"95% CI: [{ci['lower_bound']:.4f}, {ci['upper_bound']:.4f}]")
    
    # Generate plots
    plot_accuracy_comparison(results_by_prompt, model_name, dataset_name)
    plot_error_distribution(all_results, model_name, dataset_name)
    
    # For CoT variants, analyze rationale length
    cot_results = [r for r in all_results if r.get("prompt_type") in ["cot", "self_consistency", "self_reflection"]]
    if cot_results:
        plot_rationale_length(cot_results, model_name, dataset_name)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run reasoning experiments")
    parser.add_argument("--model", type=str, choices=["gpt-4", "deepseek-r1"], required=True,
                        help="Model to use for inference")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "strategyqa"], required=True,
                        help="Dataset to evaluate on")
    parser.add_argument("--prompts", type=str, nargs="+", 
                        choices=["zero_shot", "few_shot", "cot", "self_consistency", "self_reflection", "react", "all"],
                        default=["all"],
                        help="Prompt types to evaluate")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process")
    
    args = parser.parse_args()
    
    # Expand "all" to all prompt types
    prompt_types = args.prompts
    if "all" in prompt_types:
        prompt_types = ["zero_shot", "few_shot", "cot", "self_consistency", "self_reflection", "react"]
    
    # Run the experiment
    run_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        prompt_types=prompt_types,
        max_examples=args.max_examples
    )


if __name__ == "__main__":
    main() 