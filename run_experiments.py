"""
Main script to run all experiments.
"""
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

import pandas as pd
import numpy as np
import scipy.stats as st

from src.prompts import get_prompt_template
from src.inference import get_model_interface, run_inference
from src.eval import evaluate_gsm8k as evaluate_gsm8k_main
from src.eval import evaluate_strategyqa as evaluate_strategyqa_main
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


# Helper for robust number extraction (simplified)
def _extract_number_robust(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    text_str = str(text).strip()
    text_str = text_str.replace(",", "") # Remove commas like 1,000
    
    # Check for #### pattern and take the part after it
    if "####" in text_str:
        text_str = text_str.split("####")[-1].strip()

    # Try to find a number in the remaining string
    # This regex matches integers and decimals, optionally signed.
    # It will extract the first number found.
    match = re.search(r"[-+]?\d*\.?\d+", text_str)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None

def _is_correct_gsm8k_robust(result_item: Dict[str, Any]) -> bool:
    true_answer_str = result_item.get("true_answer")
    
    predicted_answer_str = None
    if result_item.get("prompt_type") == "self_reflection" and "final_answer" in result_item:
        predicted_answer_str = result_item.get("final_answer")
    else:
        predicted_answer_str = result_item.get("answer")

    if predicted_answer_str is None or true_answer_str is None:
        return False

    pred_num = _extract_number_robust(predicted_answer_str)
    true_num = _extract_number_robust(true_answer_str)

    if pred_num is not None and true_num is not None:
        return abs(pred_num - true_num) < 1e-6 # Tolerance for float comparison
    return False

def _is_correct_strategyqa_simple(result_item: Dict[str, Any]) -> bool:
    true_answer = str(result_item.get("true_answer", "")).lower().strip()
    
    predicted_answer_str = None
    if result_item.get("prompt_type") == "self_reflection" and "final_answer" in result_item:
        predicted_answer_str = result_item.get("final_answer")
    else:
        predicted_answer_str = result_item.get("answer")
        
    predicted_answer = str(predicted_answer_str if predicted_answer_str is not None else "").lower().strip()
    return predicted_answer == true_answer


# Modified accuracy calculation function to be dataset-aware
def calculate_accuracy(results: List[Dict[str, Any]], dataset_name: str) -> float:
    if not results:
        return 0.0
    
    correct = 0
    for result in results:
        is_correct_val = False
        if dataset_name == "gsm8k":
            is_correct_val = _is_correct_gsm8k_robust(result)
        elif dataset_name == "strategyqa":
            is_correct_val = _is_correct_strategyqa_simple(result)
        else: # Fallback for unknown datasets (maintains previous simple logic)
            if result.get("prompt_type") == "self_reflection" and "final_answer" in result:
                is_correct_val = result.get("final_answer") == result.get("true_answer")
            else:
                is_correct_val = result.get("answer") == result.get("true_answer")
        
        if is_correct_val:
            correct += 1
    
    return correct / len(results) if results else 0.0


# Fixed bootstrap CI wrapper, now takes dataset_name
def get_bootstrap_ci(results: List[Dict[str, Any]], dataset_name: str, confidence: float = 0.95, n_bootstrap: int = 1000):
    """Get bootstrap confidence interval for accuracy."""
    if not results: # If there are no results, accuracy is 0 and CI is [0,0]
        return {"estimate": 0.0, "lower_bound": 0.0, "upper_bound": 0.0}
    
    original_acc = calculate_accuracy(results, dataset_name)
    n = len(results)
    
    # For small sample sizes (n < 30), use Wilson score interval
    # This will now correctly handle n=1, 2, 3, 4 etc. up to 29
    if n < 30:
        k = int(round(original_acc * n))
        
        # Ensure k is within valid bounds [0, n]
        k = max(0, min(n, k))

        ci_low, ci_high = st.binomtest(k, n).proportion_ci(
            confidence_level=confidence, method="wilson"
        )
        
        return {
            "estimate": original_acc,
            "lower_bound": ci_low,
            "upper_bound": ci_high
        }
    
    # For larger samples (n >= 30), use bootstrap
    accuracies = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        bootstrap_sample = [results[i] for i in indices]
        acc = calculate_accuracy(bootstrap_sample, dataset_name) # Pass dataset_name
        accuracies.append(acc)
    
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(accuracies, alpha * 100)
    upper_bound = np.percentile(accuracies, (1 - alpha) * 100)
    
    return {
        "estimate": original_acc,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }


def run_experiment(
    model_name: str,
    dataset_name: str,
    prompt_types: List[str],
    max_examples: int = None,
    save_intermediate: bool = True,
    temperature: float = 0.0
):
    """
    Run a complete experiment for a model on a dataset with multiple prompt types.
    
    Args:
        model_name: Name of the model (gpt-4, deepseek-r1)
        dataset_name: Name of the dataset (gsm8k, strategyqa)
        prompt_types: List of prompt types to evaluate
        max_examples: Maximum number of examples to process
        save_intermediate: Whether to save intermediate results
        temperature: Temperature for model generation
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
            batch_size=5,  # Save every 5 examples
            temperature=temperature
        )
        
        # Special post-processing for self-reflection to ensure we use the correct answer
        if prompt_type == "self_reflection":
            for result in results:
                if "final_answer" in result:
                    # Make sure the final_answer after reflection is also in answer
                    result["answer"] = result["final_answer"]
        
        # Save results
        save_results(results, model_name, dataset_name, prompt_type)
        all_results.extend(results)
    
    # Run evaluation based on dataset
    if dataset_name == "gsm8k":
        metrics = evaluate_gsm8k_main(all_results)
    elif dataset_name == "strategyqa":
        metrics = evaluate_strategyqa_main(all_results)
    else: # Fallback if dataset is neither
        metrics = {"total": len(all_results), "accuracy": calculate_accuracy(all_results, dataset_name)}
    
    # Generate evaluation report
    print("\n=== Evaluation Results ===")
    print(f"Model: {model_name}, Dataset: {dataset_name}")
    print(f"Total examples: {metrics['total']}")
    print(f"Overall accuracy: {metrics['accuracy']:.4f}")
    
    # Compare prompt types
    results_by_prompt = {}
    for prompt_type in prompt_types:
        prompt_results = [r for r in all_results if r.get("prompt_type") == prompt_type]
        
        # Special handling for self-reflection to ensure using final_answer
        if prompt_type == "self_reflection":
            for result in prompt_results:
                if "final_answer" in result and not result.get("answer"): # Should be redundant if earlier processing was done
                    result["answer"] = result["final_answer"]
        
        current_prompt_metrics = {}
        if dataset_name == "gsm8k":
            current_prompt_metrics = evaluate_gsm8k_main(prompt_results)
        elif dataset_name == "strategyqa":
            current_prompt_metrics = evaluate_strategyqa_main(prompt_results)
        else: # Fallback
            current_prompt_metrics = {"accuracy": calculate_accuracy(prompt_results, dataset_name), 
                                      "total": len(prompt_results)}

        results_by_prompt[prompt_type] = current_prompt_metrics
        # Ensure accuracy key exists before printing
        accuracy_to_print = current_prompt_metrics.get('accuracy', 0.0)
        print(f"\n{prompt_type} accuracy: {accuracy_to_print:.4f}")
        
        # Get confidence intervals using the fixed bootstrap method, passing dataset_name
        ci = get_bootstrap_ci(prompt_results, dataset_name)
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
    parser = argparse.ArgumentParser(description="Run experiments with different reasoning strategies.")
    parser.add_argument("--model", required=True, choices=["gpt-4", "deepseek-r1"], help="Model to use")
    parser.add_argument("--dataset", required=True, choices=["gsm8k", "strategyqa"], help="Dataset to use")
    parser.add_argument("--prompts", nargs="+", default=["zero_shot"], 
                      choices=["zero_shot", "few_shot", "cot", "self_consistency", "self_reflection", "react", "all"],
                      help="Prompt templates to use")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to process")
    
    # Add these new arguments:
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for model generation")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference and only evaluate existing results")
    
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
        max_examples=args.max_examples,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main() 