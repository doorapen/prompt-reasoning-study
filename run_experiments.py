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
from tqdm import tqdm

from src.prompts import get_prompt_template, PROMPT_TEMPLATE_CLASSES, ALL_FEW_SHOT_EXAMPLES
from src.inference import get_model_interface, GPT4Interface, DeepSeekInterface, run_inference
from src.eval import evaluate_gsm8k as evaluate_gsm8k_main
from src.eval import evaluate_strategyqa as evaluate_strategyqa_main
from src.plots import plot_accuracy_comparison, plot_error_distribution, plot_rationale_length

# Make sure Path is imported from pathlib
from pathlib import Path
import random # Added for potential future use? Already present in prompts.py


def load_dataset(dataset_name: str, split: str = "test", max_examples: int = None) -> List[Dict[str, Any]]:
    """
    Load a dataset for evaluation.
    
    Args:
        dataset_name: Name of the dataset (gsm8k, strategyqa, math, commonsenseqa)
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
        
        # First check if files exist and have proper size
        strategyqa_dir = os.path.join("data", dataset_name)
        strategyqa_files = {
            "train": os.path.join(strategyqa_dir, "strategyqa_train.json"), 
            "test": os.path.join(strategyqa_dir, "strategyqa_test.json")
        }
        
        # Check if files exist and are valid (more than 1000 bytes)
        # Ensure the split being requested is checked if it exists in strategyqa_files
        file_to_check = strategyqa_files.get(split)
        file_exists_and_valid = False
        if file_to_check:
            file_exists_and_valid = (
                os.path.exists(file_to_check) and 
                os.path.getsize(file_to_check) > 1000
            )
        
        examples = [] # Initialize examples

        if not file_exists_and_valid:
            print(f"StrategyQA dataset file {file_to_check if file_to_check else 'for split '+split} missing or invalid. Downloading via HuggingFace...")
            try:
                # Corrected dataset name to the working ChilleD/StrategyQA
                dataset_hf = load_dataset("ChilleD/StrategyQA")
                examples = list(dataset_hf[split]) # Convert to list
                
                # Save files for future use
                # Ensure the directory exists before saving
                os.makedirs(strategyqa_dir, exist_ok=True)

                # Save the specific split downloaded
                if file_to_check: # Only save if a path was defined for this split
                    print(f"Saving downloaded {split} split to {file_to_check}")
                    with open(file_to_check, 'w') as f:
                        json.dump(examples, f)
                
                # Also save other splits if they are part of the downloaded dataset and not yet saved
                for s_key, s_path in strategyqa_files.items():
                    if s_key in dataset_hf and (not os.path.exists(s_path) or os.path.getsize(s_path) <= 1000):
                        print(f"Saving {s_key} split to {s_path}")
                        with open(s_path, 'w') as f:
                            json.dump(list(dataset_hf[s_key]), f) # Convert to list before dumping

            except Exception as e:
                print(f"Error downloading StrategyQA dataset: {e}")
                # Try using the already created file for the specific split if possible (even if small)
                if file_to_check and os.path.exists(file_to_check) and os.path.getsize(file_to_check) > 0:
                    print(f"Using existing file as fallback: {file_to_check}")
                    try:
                        with open(file_to_check, 'r') as f:
                            examples = json.load(f)
                    except Exception as e2:
                        print(f"Error loading local file {file_to_check}: {e2}")
                        examples = []
                else:
                    print("No valid local files found. Using fallback options (empty list)...")
                    examples = []
        else:
            print(f"Using existing StrategyQA file: {file_to_check}")
            try:
                with open(file_to_check, 'r') as f:
                    examples = json.load(f)
                print(f"Loaded {len(examples)} examples from local file {file_to_check}")
            except Exception as e2:
                print(f"Error loading local StrategyQA file {file_to_check}: {e2}")
                # Fallback to HuggingFace if local load fails despite file existing
                print("Attempting to download from HuggingFace as fallback...")
                try:
                    dataset_hf = load_dataset("ChilleD/StrategyQA")
                    examples = list(dataset_hf[split])
                    # Attempt to save it again if it was corrupted
                    if file_to_check:
                        print(f"Re-saving downloaded {split} split to {file_to_check}")
                        with open(file_to_check, 'w') as f:
                            json.dump(examples, f)
                except Exception as e_hf_fallback:
                    print(f"Error downloading StrategyQA from HuggingFace as fallback: {e_hf_fallback}")
                    examples = []
        
        # Convert to list of dictionaries
        data = []
        for i, ex in enumerate(examples):
            if max_examples and i >= max_examples:
                break
            try:
                # Handle both dictionary and dataset formats
                question = ex.get("question", "") if isinstance(ex, dict) else ex["question"]
                answer_value = ex.get("answer", False) if isinstance(ex, dict) else ex["answer"]
                answer_text = "yes" if answer_value in [True, "yes", "Yes", 1, "1"] else "no"
                
                data.append({
                    "id": f"strategyqa_{split}_{i}",
                    "question": question,
                    "answer": answer_text
                })
            except (KeyError, AttributeError, TypeError) as e:
                print(f"Error processing StrategyQA example {i}: {ex} - Error: {e}")
                # Continue to next example rather than breaking
                continue
        
    elif dataset_name == "math":
        # Check both directory structure and JSONL format
        math_dir_path = os.path.join("data", dataset_name, split)
        math_file_path = os.path.join("data", dataset_name, f"{split}.jsonl")
        
        data = []
        
        if os.path.exists(math_dir_path) and os.path.isdir(math_dir_path):
            # Directory structure found (with subject folders like algebra, geometry)
            print(f"Loading MATH dataset from directory structure: {math_dir_path}")
            import glob
            
            # Find all JSON files in subdirectories
            json_pattern = os.path.join(math_dir_path, "**", "*.json")
            json_files = glob.glob(json_pattern, recursive=True)
            
            for i, file_path in enumerate(json_files):
                if max_examples and i >= max_examples:
                    break
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        item = json.load(f)
                        
                        # Extract true answer, often in \boxed{}
                        true_answer_boxed = None
                        if isinstance(item.get("solution"), str):
                            match_boxed = re.search(r"\\boxed{(.+?)}", item["solution"])
                            if match_boxed:
                                true_answer_boxed = match_boxed.group(1).strip()
                            else:
                                # Fallback if no box, could be the whole solution or last part
                                true_answer_boxed = item["solution"].strip()
                        
                        data.append({
                            "id": f"math_{split}_{os.path.basename(file_path)}",
                            "question": item.get("problem", ""),
                            "answer": true_answer_boxed,
                            "level": item.get("level"),
                            "type": item.get("type")
                        })
                except Exception as e:
                    print(f"Error loading MATH file {file_path}: {e}")
        
        elif os.path.exists(math_file_path):
            # Single JSONL file format
            print(f"Loading MATH dataset from JSONL file: {math_file_path}")
            try:
                with open(math_file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if max_examples and i >= max_examples:
                            break
                        item = json.loads(line)
                        
                        # Extract true answer, often in \boxed{}
                        true_answer_boxed = None
                        if isinstance(item.get("solution"), str):
                            match_boxed = re.search(r"\\boxed{(.+?)}", item["solution"])
                            if match_boxed:
                                true_answer_boxed = match_boxed.group(1).strip()
                            else:
                                true_answer_boxed = item["solution"].strip()
                        
                        data.append({
                            "id": f"math_{split}_{i}",
                            "question": item.get("problem", ""),
                            "answer": true_answer_boxed,
                            "level": item.get("level"),
                            "type": item.get("type")
                        })
            except Exception as e:
                print(f"Error loading MATH dataset from {math_file_path}: {e}")
        
        else:
            print(f"Warning: MATH dataset not found at {math_dir_path} or {math_file_path}. Returning empty list.")
        
    elif dataset_name == "commonsenseqa":
        # Try loading from files first, then fall back to HuggingFace
        from datasets import load_dataset
        
        csqa_dir = os.path.join("data", dataset_name)
        csqa_files = {
            "train": os.path.join(csqa_dir, "commonsenseqa_train.jsonl"),
            "dev": os.path.join(csqa_dir, "commonsenseqa_dev.jsonl"),
            "test": os.path.join(csqa_dir, "commonsenseqa_test.jsonl")
        }
        
        # Map HuggingFace split names to file split names
        hf_to_file_split = {"train": "train", "validation": "dev", "test": "test"}
        file_split = hf_to_file_split.get(split, split)
        
        # Check if files exist and are valid (more than 1000 bytes)
        file_exists_and_valid = (
            os.path.exists(csqa_files[file_split]) and
            os.path.getsize(csqa_files[file_split]) > 100
        )
        
        data = []
        
        if file_exists_and_valid:
            try:
                with open(csqa_files[file_split], 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if max_examples and i >= max_examples:
                            break
                        try:
                            item = json.loads(line)
                            question_text = item.get("question", {}).get("stem", "")
                            choices_data = item.get("question", {}).get("choices", []) # Use this for choices
                            
                            # If choices_data is empty, try to get it from top-level "choices"
                            if not choices_data and "choices" in item:
                                choices_data = item.get("choices", [])

                            choices_text_parts = []
                            for c in choices_data:
                                label = c.get('label', '')
                                text = c.get('text', '')
                                if label and text: # Ensure both label and text exist
                                     choices_text_parts.append(f"({label}) {text}")
                            
                            choices_text_str = ""
                            if choices_text_parts: # Only add choices if they were successfully parsed
                                choices_text_str = "\\nChoices:\\n" + "\\n".join(choices_text_parts)
                            
                            raw_answer_key = item.get("answerKey")
                            data.append({
                                "id": f"csqa_{split}_{i}",
                                "question": question_text + choices_text_str,
                                "answer": str(raw_answer_key).upper() if raw_answer_key is not None else ""
                            })
                        except json.JSONDecodeError:
                            print(f"Warning: Invalid JSON on line {i+1} in {csqa_files[file_split]}")
            except Exception as e:
                print(f"Error loading CommonsenseQA from local file {csqa_files[file_split]}: {e}")
                data = []

        if not data: # If local loading failed or file was not valid
            print(f"Local CommonsenseQA data for split '{file_split}' not found or failed to load. Attempting HuggingFace download.")
            try:
                # Try the more standard HuggingFace dataset name
                dataset_hf = load_dataset("commonsense_qa")
                actual_hf_split = "validation" if split == "test" else split # HF uses "validation" for test
                
                if actual_hf_split not in dataset_hf:
                    print(f"Split '{actual_hf_split}' not found in HuggingFace dataset commonsense_qa. Available: {list(dataset_hf.keys())}")
                    raise KeyError(f"Split '{actual_hf_split}' not available.")

                examples_hf = dataset_hf[actual_hf_split]
                
                temp_data_for_saving = [] # Store structured data for saving
                
                for i, ex_hf in enumerate(examples_hf):
                    if max_examples and len(data) >= max_examples: # Check against len(data)
                        break
                    
                    question_text = ex_hf["question"]
                    choices_list_hf = ex_hf["choices"]["text"]
                    choice_labels_hf = ex_hf["choices"]["label"] # Usually A, B, C, D, E
                    raw_answer_key_hf = ex_hf.get("answerKey") 

                    choices_for_prompt = []
                    choices_for_saving = [] # For saving in original JSONL format

                    for label_char, choice_text_hf in zip(choice_labels_hf, choices_list_hf):
                        choices_for_prompt.append(f"({label_char}) {choice_text_hf}")
                        choices_for_saving.append({"label": label_char, "text": choice_text_hf})
                    
                    choices_text_str_hf = "\\nChoices:\\n" + "\\n".join(choices_for_prompt)
                    
                    processed_answer_hf = str(raw_answer_key_hf).upper() if raw_answer_key_hf is not None else ""
                    
                    data.append({
                        "id": f"csqa_{split}_{i}", # Use original requested split name for ID
                        "question": question_text + choices_text_str_hf,
                        "answer": processed_answer_hf
                    })

                    # Prepare item for saving in a format compatible with local loader
                    temp_data_for_saving.append({
                        "id": f"csqa_{split}_{i}", # For consistency, though not used by local loader
                        "question": {"stem": question_text, "choices": choices_for_saving},
                        "answerKey": raw_answer_key_hf # Save the original key
                    })
                
                if temp_data_for_saving: # If we successfully processed and formatted data
                    os.makedirs(csqa_dir, exist_ok=True)
                    target_save_file = csqa_files[file_split]
                    print(f"Attempting to save {len(temp_data_for_saving)} downloaded CommonsenseQA items to {target_save_file}")
                    try:
                        with open(target_save_file, 'w', encoding='utf-8') as f:
                            for item_to_save in temp_data_for_saving:
                                f.write(json.dumps(item_to_save) + "\n")
                        print(f"Finished saving downloaded CommonsenseQA data to {target_save_file}")
                    except Exception as e_save:
                        print(f"Error saving downloaded CommonsenseQA data to {target_save_file}: {e_save}")
            
            except Exception as e_hf:
                print(f"Error downloading or processing CommonsenseQA from HuggingFace: {e_hf}")
                import traceback
                traceback.print_exc()
                # If HuggingFace also fails, and data is still empty, it means all attempts failed.
                if not data:
                    print(f"All attempts to load CommonsenseQA for split '{split}' failed.")
                    return [] # Return empty list as per original logic for MATH dataset not found
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return data


def save_results(results: List[Dict[str, Any]], 
                model_name: str, 
                dataset_name: str, 
                prompt_type: str):
    """Save results to a JSON file."""
    results_dir = Path("results") / dataset_name / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prompt_type}_{timestamp}.json"
    
    with open(results_dir / filename, "w") as f:
        json.dump(results, f, indent=2)


def save_batch(batch: List[Dict[str, Any]], 
              model_name: str, 
              dataset_name: str, 
              prompt_type: str):
    """Save a batch of results (used as a callback during inference)."""
    results_dir = Path("results") / dataset_name / model_name / prompt_type
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Use first item's ID for the batch file
    batch_id = batch[0]["id"]
    filename = f"batch_{batch_id}.json"
    
    with open(results_dir / filename, "w") as f:
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

def _is_correct_gsm8k_robust(result_item: Dict[str, Any], original_item: Dict[str, Any]) -> bool:
    true_answer_str = original_item.get("answer")
    
    predicted_answer_str = None
    if result_item.get("prompt_type") == "self_reflection" and "final_answer_from_reflection" in result_item:
        predicted_answer_str = result_item.get("final_answer_from_reflection")
    elif result_item.get("prompt_type") == "self_reflection" and "final_answer" in result_item:
        predicted_answer_str = result_item.get("final_answer")
    else:
        predicted_answer_str = result_item.get("answer")

    if predicted_answer_str is None or true_answer_str is None:
        return False

    pred_num = None
    if isinstance(predicted_answer_str, (float, int)):
        pred_num = float(predicted_answer_str)
    else:
        pred_num = _extract_number_robust(str(predicted_answer_str))
        
    true_num = _extract_number_robust(str(true_answer_str))

    if pred_num is not None and true_num is not None:
        return abs(pred_num - true_num) < 1e-6
    return False

def _is_correct_strategyqa(result_item: Dict[str, Any], original_item: Dict[str, Any]) -> bool:
    true_answer_str = str(original_item.get("answer", "")).lower().strip()
    true_answer_bool = None
    if true_answer_str == "yes":
        true_answer_bool = True
    elif true_answer_str == "no":
        true_answer_bool = False
    
    predicted_answer_val = None
    if result_item.get("prompt_type") == "self_reflection" and "final_answer_from_reflection" in result_item:
        predicted_answer_val = result_item.get("final_answer_from_reflection")
    elif result_item.get("prompt_type") == "self_reflection" and "final_answer" in result_item:
        predicted_answer_val = result_item.get("final_answer")
    else:
        predicted_answer_val = result_item.get("answer")

    if predicted_answer_val is None or true_answer_bool is None:
        return False

    if isinstance(predicted_answer_val, bool):
        return predicted_answer_val == true_answer_bool
    
    if isinstance(predicted_answer_val, str):
        pred_str_lower = predicted_answer_val.lower().strip()
        # More robust check for "yes"/"no" and "true"/"false"
        if pred_str_lower in ["yes", "true"]: 
            return true_answer_bool is True
        if pred_str_lower in ["no", "false"]: 
            return true_answer_bool is False
            
    return False

def _is_correct_math(result_item: Dict[str, Any], original_item: Dict[str, Any]) -> bool:
    """
    Evaluates if the predicted MATH answer is correct.
    IMPORTANT: This is a VERY basic placeholder. True MATH evaluation requires
    symbolic comparison of LaTeX expressions (e.g., using sympy).
    """
    true_answer_latex = original_item.get("answer") # Ground truth LaTeX string
    
    predicted_answer_val = None
    if result_item.get("prompt_type") == "self_reflection" and "final_answer_from_reflection" in result_item:
        predicted_answer_val = result_item.get("final_answer_from_reflection")
    elif result_item.get("prompt_type") == "self_reflection" and "final_answer" in result_item:
        predicted_answer_val = result_item.get("final_answer")
    else:
        predicted_answer_val = result_item.get("answer") # Parsed LaTeX string from model

    if predicted_answer_val is None or true_answer_latex is None:
        return False

    # Basic normalization (extremely simplified, not robust for LaTeX)
    norm_true = str(true_answer_latex).strip().replace(" ", "").lower()
    norm_pred = str(predicted_answer_val).strip().replace(" ", "").lower()
    
    # Remove common LaTeX commands that might differ but mean the same for simple cases
    # This is still very heuristic.
    for cmd in ["\\frac", "\\cdot", "\\left", "\\right", "(", ")", "{", "}"]:
        norm_true = norm_true.replace(cmd, "")
        norm_pred = norm_pred.replace(cmd, "")

    return norm_true == norm_pred

def _is_correct_commonsenseqa(result_item: Dict[str, Any], original_item: Dict[str, Any]) -> bool:
    """
    Evaluates if the predicted CommonsenseQA answer (choice letter) is correct.
    """
    true_answer_label = original_item.get("answer") # Ground truth label (e.g., "A")
    
    predicted_answer_val = None
    if result_item.get("prompt_type") == "self_reflection" and "final_answer_from_reflection" in result_item:
        predicted_answer_val = result_item.get("final_answer_from_reflection")
    elif result_item.get("prompt_type") == "self_reflection" and "final_answer" in result_item:
        predicted_answer_val = result_item.get("final_answer")
    else:
        predicted_answer_val = result_item.get("answer") # Parsed choice letter from model

    if predicted_answer_val is None or true_answer_label is None:
        return False

    # Predicted answer should be a single uppercase letter string
    if isinstance(predicted_answer_val, str):
        # Handle cases like "A" or "A." or "(A)"
        # Extract first uppercase letter found
        match = re.search(r"([A-E])", predicted_answer_val.strip().upper())
        if match:
            return match.group(1) == true_answer_label.strip().upper()
    return False


# Define a dictionary for evaluation functions
EVALUATION_FUNCTIONS = {
    "gsm8k": _is_correct_gsm8k_robust,
    "strategyqa": _is_correct_strategyqa,
    "math": _is_correct_math,
    "commonsenseqa": _is_correct_commonsenseqa,
}


def calculate_accuracy(results: List[Dict[str, Any]], dataset_name: str, original_dataset: List[Dict[str,Any]]) -> float:
    """Calculate accuracy for a list of results."""
    if not results:
        return 0.0
    
    if dataset_name not in EVALUATION_FUNCTIONS:
        print(f"Warning: No evaluation function for {dataset_name} in calculate_accuracy. Returning 0.0.")
        return 0.0
    eval_func = EVALUATION_FUNCTIONS[dataset_name]

    # Create a map for original items for quick lookup
    original_items_map = {item['id']: item for item in original_dataset}

    correct_predictions = 0
    evaluable_count = 0
    for res_item in results:
        original_item = original_items_map.get(res_item['id'])
        if not original_item:
            # print(f"Warning: Original item not found for result ID {res_item.get('id')}. Skipping.")
            continue

        # Check if the item is evaluable (has an answer field from parsing)
        # and a corresponding true answer
        has_predicted_answer = False
        if res_item.get("prompt_type") == "self_reflection":
            has_predicted_answer = (res_item.get("final_answer_from_reflection") is not None or 
                                    res_item.get("final_answer") is not None)
        else:
            has_predicted_answer = res_item.get("answer") is not None
        
        has_true_answer = original_item.get("answer") is not None
        
        if has_predicted_answer and has_true_answer:
            evaluable_count += 1
            if eval_func(res_item, original_item):
                correct_predictions += 1
        # else:
            # print(f"Item {res_item.get('id')} not evaluable. Predicted: {has_predicted_answer}, True: {has_true_answer}")
                
    return correct_predictions / evaluable_count if evaluable_count > 0 else 0.0


def get_bootstrap_ci(results: List[Dict[str, Any]], dataset_name: str, original_dataset: List[Dict[str,Any]], n_bootstrap: int = 1000, alpha: float = 0.05) -> Dict[str, float]:
    """Calculate bootstrap confidence interval for accuracy."""
    if not results:
        return {"lower_bound": 0.0, "upper_bound": 0.0, "mean_accuracy": 0.0}

    if dataset_name not in EVALUATION_FUNCTIONS:
        print(f"Warning: No evaluation function for {dataset_name} in get_bootstrap_ci. Returning 0 CI.")
        return {"lower_bound": 0.0, "upper_bound": 0.0, "mean_accuracy": 0.0}
    
    # Filter results to only those that can be evaluated against an original item
    # and have a predicted answer.
    original_items_map = {item['id']: item for item in original_dataset}
    evaluable_results_with_original = []
    for r in results:
        original_item = original_items_map.get(r['id'])
        if not original_item or original_item.get("answer") is None:
            continue # Skip if no original item or original item has no answer

        has_predicted_answer = False
        if r.get("prompt_type") == "self_reflection":
            has_predicted_answer = (r.get("final_answer_from_reflection") is not None or 
                                    r.get("final_answer") is not None)
        else:
            has_predicted_answer = r.get("answer") is not None
        
        if has_predicted_answer:
            evaluable_results_with_original.append(r)


    if not evaluable_results_with_original:
        return {"lower_bound": 0.0, "upper_bound": 0.0, "mean_accuracy": 0.0}

    accuracies = []
    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping CI", leave=False, ncols=80):
        # Sample with replacement from the *indices* of evaluable_results_with_original
        # This ensures that when we calculate accuracy, we use the *same* original_dataset
        # for looking up true answers, but the sample of *predictions* changes.
        
        # We are bootstrapping the set of predictions.
        # The `calculate_accuracy` function needs the original_dataset to fetch true answers.
        bootstrap_indices = np.random.choice(len(evaluable_results_with_original), size=len(evaluable_results_with_original), replace=True)
        bootstrap_sample_predictions = [evaluable_results_with_original[i] for i in bootstrap_indices]
        
        sample_accuracy = calculate_accuracy(bootstrap_sample_predictions, dataset_name, original_dataset)
        accuracies.append(sample_accuracy)
    
    lower_bound = np.percentile(accuracies, (alpha / 2) * 100)
    upper_bound = np.percentile(accuracies, (1 - alpha / 2) * 100)
    
    return {"lower_bound": lower_bound, "upper_bound": upper_bound, "mean_accuracy": np.mean(accuracies)}


def run_experiment(
    model_name: str,
    dataset_name: str,
    prompt_types: List[str],
    max_examples: int = None,
    save_intermediate_batches: bool = False,
    batch_size_intermediate: int = 20,
    temperature: float = 0.0
):
    """
    Run a complete experiment for a model on a dataset with multiple prompt types.
    Refactored to use the run_inference utility.
    """
    # Load test dataset
    test_data = load_dataset(dataset_name, split="test", max_examples=max_examples)
    if not test_data:
        print(f"Error: Failed to load test data for {dataset_name}. Skipping.")
        return
    print(f"Loaded {len(test_data)} test examples from {dataset_name}")
    
    # Load train dataset *only* if needed for few-shot on MATH
    train_data = None
    if dataset_name == "math" and any(pt in ["few_shot", "few_shot_cot"] for pt in prompt_types): # Adjusted for new prompt types
        print(f"Loading train data for {dataset_name} (for few-shot type matching)...")
        train_data = load_dataset(dataset_name, split="train", max_examples=None) 
        if not train_data:
             print(f"Warning: Failed to load train data for {dataset_name}. Few-shot type matching will be disabled.")
        else:
             print(f"Loaded {len(train_data)} train examples for {dataset_name}.")
    
    # Create a main results directory structure
    main_results_dir = Path("results") / dataset_name
    main_results_dir.mkdir(parents=True, exist_ok=True)

    # Get model interface
    try:
        model_interface = get_model_interface(model_name, temperature=temperature) # Pass temperature here
    except Exception as e:
        print(f"Error initializing model {model_name}: {e}. Aborting experiment.")
        return
    
    all_results_for_dataset = [] # This will store results from *all* prompt types for this run
    
    # Run inference for each prompt type using the run_inference utility
    for prompt_type in prompt_types:
        print(f"\nRunning {model_name} on {dataset_name} with {prompt_type} prompting (Temp: {temperature})")
        
        # Prepare results directory and filename for this specific run
        # Using .jsonl for streaming results via callback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace('/', '_').replace(':', '_')
        # Include temperature in filename for clarity
        results_filename = f"{safe_model_name}_{prompt_type}_temp{str(temperature).replace('.', 'p')}_{timestamp}.jsonl"
        results_filepath = main_results_dir / results_filename
        
        # Clear the file if it exists before starting a new run
        try:
            with open(results_filepath, 'w') as f:
                pass 
            print(f"   Cleared/prepared results file: {results_filepath}")
        except IOError as e:
             print(f"   Error preparing results file {results_filepath}: {e}. Skipping prompt type {prompt_type}.")
             continue

        try:
            # Prepare prompt template, passing train_data if needed for few_shot MATH
            template_kwargs = {}
            if prompt_type in ["few_shot", "few_shot_cot"] and dataset_name == "math":
                if train_data:
                    template_kwargs["train_dataset"] = train_data
                else:
                    print("Warning: Running few-shot on MATH but train_data unavailable. Type matching disabled.")

            # Add specific arguments for certain prompt types (passed to template constructor)
            if prompt_type == "self_consistency":
                template_kwargs["num_samples"] = 5 # Example value, make configurable if needed
            
            # Example: if few_shot needs a specific number of examples different from default
            # if prompt_type in ["few_shot", "few_shot_cot"]:
            #     template_kwargs["num_examples"] = 3 # Default is 3, can be overridden

            prompt_template_instance = get_prompt_template(
                template_name=prompt_type, 
                dataset_name=dataset_name,
                **template_kwargs
            )

            # Define batch callback for saving intermediate results
            def save_batch_results(batch_results):
                try:
                    with open(results_filepath, 'a', encoding='utf-8') as f:
                        for result in batch_results:
                            # Ensure result is serializable (handles numpy types)
                            serializable_result = {
                                k: (str(v) if isinstance(v, (np.int64, np.float64, np.bool_)) else v) 
                                for k, v in result.items()
                            }
                            try:
                                 f.write(json.dumps(serializable_result) + '\n')
                            except TypeError as te:
                                 print(f"Serialization Error: {te} for result: {serializable_result}")
                                 # Fallback for unhandled types: convert all to string
                                 fallback_serializable = {k: str(v) for k,v in result.items()}
                                 try:
                                     f.write(json.dumps(fallback_serializable) + '\n')
                                 except Exception as fe:
                                     print(f"Fallback Serialization Error: {fe}")
                                     f.write(json.dumps({"id": result.get("id"), "error": "Serialization failed completely"}) + '\n')
                except IOError as e_io:
                    print(f"Error writing batch results to {results_filepath}: {e_io}")

            # Use the run_inference function
            print(f"   Starting inference using run_inference utility...")
            current_prompt_type_results = run_inference(
                model=model_interface, # Model interface already has temperature set
                prompt_template=prompt_template_instance,
                dataset=test_data, # The (potentially max_examples limited) test data for this run
                examples=None, # Few-shot examples handled internally by FewShotPrompt types
                # temperature is now set in model_interface
                max_tokens=1024 if dataset_name != "math" else 2048, # More tokens for MATH
                batch_callback=save_batch_results if save_intermediate_batches else None,
                batch_size=batch_size_intermediate 
            )
            
            print(f"   Completed inference for {prompt_type}. {len(current_prompt_type_results)} results obtained (some may be errors). Results streamed to {results_filepath}")
            # Add prompt_type to each result dict for easier filtering later
            for res in current_prompt_type_results:
                res["prompt_type"] = prompt_type 
                res["model_name"] = model_name # Add model name for comprehensive results
                res["dataset_name"] = dataset_name # Add dataset name
            all_results_for_dataset.extend(current_prompt_type_results)

        except Exception as e_inf:
            print(f"   Error running inference for {prompt_type}: {e_inf}")
            import traceback
            traceback.print_exc()
        
        # --- Loop for prompt types ends here ---

    # --- Evaluation Section --- 
    # This section now evaluates based on the `all_results_for_dataset` which contains
    # results from all prompt types run in this experiment execution.
    # The `full_test_data_for_eval` is the ground truth.
    
    print("\n--- Aggregating and Evaluating All Results for this Run ---")
    # Load the original test dataset again for evaluation.
    # This ensures that even if `max_examples` was used for the inference run,
    # evaluation (especially if done separately or if `all_results_for_dataset` combines
    # multiple files) uses a consistent ground truth.
    # If `max_examples` is set, evaluation should also be on that subset for apples-to-apples.
    print(f"Loading test data for evaluation (max_examples={max_examples})...")
    full_test_data_for_eval = load_dataset(dataset_name, split="test", max_examples=max_examples)
    if not full_test_data_for_eval:
         print("Error: Failed to load test data for evaluation. Cannot calculate accuracy.")
         return 

    # Overall evaluation for all results collected in this run
    if dataset_name not in EVALUATION_FUNCTIONS:
        print(f"Error: No evaluation function defined for dataset '{dataset_name}'. Cannot calculate overall accuracy.")
        overall_metrics = {"total_processed_in_run": len(all_results_for_dataset), "accuracy": 0.0}
    else:
        overall_accuracy = calculate_accuracy(all_results_for_dataset, dataset_name, full_test_data_for_eval)
        overall_metrics = {"total_processed_in_run": len(all_results_for_dataset), "accuracy": overall_accuracy}
    
    print("\n=== Overall Evaluation Summary for this Run ===")
    print(f"Model: {model_name}, Dataset: {dataset_name}, Temperature: {temperature}")
    print(f"Total examples processed in this run (all prompt types, incl. errors): {overall_metrics['total_processed_in_run']}")
    if dataset_name in EVALUATION_FUNCTIONS:
        print(f"Overall accuracy (across all prompt types in this run): {overall_metrics['accuracy']:.4f}")
        if all_results_for_dataset: # Calculate CI only if there are results
            overall_ci = get_bootstrap_ci(all_results_for_dataset, dataset_name, full_test_data_for_eval)
            print(f"Overall 95% CI: [{overall_ci['lower_bound']:.4f}, {overall_ci['upper_bound']:.4f}] (Mean: {overall_ci['mean_accuracy']:.4f})")
    
    # Per-prompt-type evaluation
    print("\n--- Per-Prompt Type Evaluation for this Run ---")
    results_by_prompt_type_for_plot = {} # For plotting

    for p_type in prompt_types: # Iterate through the prompt types that were run
        # Filter results from `all_results_for_dataset` for this specific prompt type
        prompt_specific_results = [r for r in all_results_for_dataset if r.get("prompt_type") == p_type]
        
        num_processed_for_prompt = len(prompt_specific_results)
        current_prompt_accuracy = 0.0
        
        if dataset_name in EVALUATION_FUNCTIONS:
             if prompt_specific_results:
                 current_prompt_accuracy = calculate_accuracy(prompt_specific_results, dataset_name, full_test_data_for_eval)
             else:
                  print(f"No results found for prompt type {p_type} in this run to calculate accuracy.")
        else:
            print(f"Warning: No specific eval function for {dataset_name} for prompt type {p_type} summary.")

        results_by_prompt_type_for_plot[p_type] = {"accuracy": current_prompt_accuracy, "total": num_processed_for_prompt}
        
        print(f"\nPrompt Type: {p_type}")
        print(f"  Accuracy: {current_prompt_accuracy:.4f} (on {num_processed_for_prompt} processed examples)")
        
        if prompt_specific_results and dataset_name in EVALUATION_FUNCTIONS:
             ci = get_bootstrap_ci(prompt_specific_results, dataset_name, full_test_data_for_eval)
             print(f"  95% CI: [{ci['lower_bound']:.4f}, {ci['upper_bound']:.4f}] (Mean: {ci['mean_accuracy']:.4f})")
        elif not prompt_specific_results:
             print("  Skipping CI calculation (no results for this prompt type in this run).")
        else: # No eval function
             print("  Skipping CI calculation (no eval function for dataset).")
    
    # Generate plots if results_by_prompt_type_for_plot is not empty
    plot_suffix = f"_temp{str(temperature).replace('.', 'p')}" # Add temperature to plot filenames
    if results_by_prompt_type_for_plot:
        plot_accuracy_comparison(results_by_prompt_type_for_plot, model_name, dataset_name, suffix=plot_suffix)
        
        if all_results_for_dataset: # Ensure there are some results to plot
            # Error distribution plot might be less meaningful if it combines all prompt types,
            # but can give a general overview.
            # plot_error_distribution(all_results_for_dataset, model_name, dataset_name, suffix=plot_suffix)
            
            # Rationale length for CoT-like prompts
            cot_like_results = [
                r for r in all_results_for_dataset 
                if r.get("prompt_type") in PROMPT_TEMPLATE_CLASSES and PROMPT_TEMPLATE_CLASSES[r.get("prompt_type")].IS_COT_LIKE
            ]
            if cot_like_results:
                plot_rationale_length(cot_like_results, model_name, dataset_name, suffix=plot_suffix)
            else:
                print("No CoT-like results found in this run for rationale length plotting.")
        else:
             print("Skipping error distribution and rationale length plots (no results in this run).")
    else:
         print("Skipping plots (no results by prompt type in this run).")
    
    print(f"\nExperiment run for {model_name} on {dataset_name} (Temp: {temperature}) with prompts {prompt_types} completed.")


def main():
    parser = argparse.ArgumentParser(description="Run experiments with different reasoning strategies.")
    parser.add_argument("--model", required=True, choices=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "deepseek-coder-6.7b-instruct", "deepseek-math-7b-instruct", "mistral-7b-instruct", "mixtral-8x7b-instruct"], help="Model to use")
    parser.add_argument("--dataset", required=True, choices=["gsm8k", "strategyqa", "math", "commonsenseqa"], help="Dataset to use")
    parser.add_argument("--prompts", nargs="+", default=["zero_shot"], 
                      choices=list(PROMPT_TEMPLATE_CLASSES.keys()) + ["all"], # Use keys from PROMPT_TEMPLATE_CLASSES
                      help="Prompt templates to use. 'all' runs all available prompts.")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to process from the dataset")
    
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for model generation")
    parser.add_argument("--save_intermediate_batches", action="store_true", help="Save intermediate batches of results during long runs.")
    parser.add_argument("--batch_size_intermediate", type=int, default=10, help="Number of items per intermediate batch save.")
    # --skip_inference argument is removed as per previous instructions to simplify focus on running experiments.
    # If evaluation of existing files is needed, a separate script/mode would be better.
    
    args = parser.parse_args()
    
    # Expand "all" to all prompt types
    prompt_types_to_run = args.prompts
    if "all" in prompt_types_to_run:
        prompt_types_to_run = list(PROMPT_TEMPLATE_CLASSES.keys())
        print(f"Running all available prompt types: {prompt_types_to_run}")
    
    # Run the experiment
    run_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        prompt_types=prompt_types_to_run,
        max_examples=args.max_examples,
        save_intermediate_batches=args.save_intermediate_batches,
        batch_size_intermediate=args.batch_size_intermediate,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main() 
