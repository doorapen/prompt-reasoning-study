import json
import os
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from scipy.stats import norm
import re
from src.prompts import (
    PROMPT_TEMPLATE_CLASSES,
    get_prompt_template,
    SelfReflectionPrompt,
    PromptTemplate,
    SelfConsistencyPrompt,
)
from datetime import datetime
import argparse
from src.plots import plot_error_distribution

# --- SymPy Import (TOP LEVEL) ---  #
SYMPY_AVAILABLE = False
try:
    from sympy import sympify, simplify, N, latex, S
    from sympy.parsing.latex import parse_latex
    # from sympy.core.compatibility import SymPyConfigError # Deprecated
    SYMPY_AVAILABLE = True
    print("INFO: SymPy imported successfully at top level.")
except ImportError as e:
    print(f"WARNING: SymPy Import Error at top level: {e}. MATH eval will use string fallback.")
except Exception as e:
    print(f"WARNING: Other error during SymPy import at top level: {e}. MATH eval will use string fallback.")
# --- End SymPy Import --- #

# --- Helper functions from run_experiments.py for consistent evaluation ---

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

def _is_correct_gsm8k_robust(result_item: Dict[str, Any], dataset_item: Dict[str, Any]) -> bool:
    true_answer_str = dataset_item.get("true_answer")
    
    predicted_answer_str = None
    if result_item.get("prompt_type") == "self_reflection" and "final_answer_from_reflection" in result_item:
        predicted_answer_str = result_item.get("final_answer_from_reflection")
    elif result_item.get("prompt_type") == "self_reflection" and "final_answer" in result_item:
        predicted_answer_str = result_item.get("final_answer")
    else:
        predicted_answer_str = result_item.get("answer")

    if predicted_answer_str is None or true_answer_str is None:
        return False

    # Use the robust extraction for both true and predicted answers
    # True answer from dataset is usually clean, but apply robust extraction for consistency
    true_num = _extract_number_robust(str(true_answer_str))
    predicted_num = _extract_number_robust(str(predicted_answer_str))

    if true_num is not None and predicted_num is not None:
        return math.isclose(true_num, predicted_num)
    
    # Fallback for non-numeric or if extraction fails (though gsm8k is numeric)
    # This part might differ if run_experiments.py had a different string fallback.
    # For gsm8k, numeric comparison is primary.
    # If numbers can't be extracted, they are not considered close.
    return False

def _is_correct_math(result_item: Dict[str, Any], dataset_item: Dict[str, Any]) -> bool:
    is_math_eval_debug = False
    # ---- YOU MANUALLY SET THIS SECTION FOR DEBUGGING ----
    # Example: Enable for a specific ID
    if result_item.get("id") == "math_test_396.json": # <-- YOUR TARGET ID HERE
        is_math_eval_debug = True
    # ---- END OF MANUAL DEBUG SECTION ----

    if SYMPY_AVAILABLE:
        try:
            true_answer_str = str(dataset_item.get("true_answer"))
            predicted_answer_val = result_item.get("answer")

            if result_item.get("prompt_type") == "self_reflection":
                if "final_answer_from_reflection" in result_item:
                    predicted_answer_val = result_item.get("final_answer_from_reflection")
                elif "final_answer" in result_item:
                    predicted_answer_val = result_item.get("final_answer")
            
            predicted_answer_str = str(predicted_answer_val)

            if is_math_eval_debug:
                print(f"-- MATH DEBUG ID (SymPy Path): {result_item.get('id')} --")
                print(f"Original True Answer (from dataset_item): {dataset_item.get('true_answer')}")
                print(f"Original Predicted Answer (from result_item.get('answer') or reflection): {predicted_answer_val}")

            if not true_answer_str or true_answer_str.lower() == "none":
                if is_math_eval_debug: print(f"True answer string is empty or None: '{true_answer_str}'")
                return False
            if not predicted_answer_str or predicted_answer_str.lower() == "none":
                if is_math_eval_debug: print(f"Predicted answer string is empty or None: '{predicted_answer_str}'")
                return False

            true_boxed_match = re.search(r"\\boxed{(.+?)}", true_answer_str)
            if true_boxed_match:
                true_answer_str = true_boxed_match.group(1)
            
            pred_boxed_match = re.search(r"\\boxed{(.+?)}", predicted_answer_str)
            if pred_boxed_match:
                predicted_answer_str = pred_boxed_match.group(1)
            
            if is_math_eval_debug:
                print(f"After boxed extraction - True: '{true_answer_str}', Pred: '{predicted_answer_str}'")

            substitutions = {
                r"\\operatorname{(\w+)}": r"\\mathrm{\1}",
                r"\\xc2\\xb0": "",
                r"\\u00b0": "",
                r"%": "/100",
                r"\\cdot": "*",
                r"\\times": "*",
                r"\\div": "/",
            }
            for pat, repl in substitutions.items():
                true_answer_str = re.sub(pat, repl, true_answer_str)
                predicted_answer_str = re.sub(pat, repl, predicted_answer_str)
            
            true_answer_str = true_answer_str.replace(" ", "").strip()
            predicted_answer_str = predicted_answer_str.replace(" ", "").strip()

            if is_math_eval_debug:
                print(f"After substitutions - True: '{true_answer_str}', Pred: '{predicted_answer_str}'")

            if not true_answer_str or true_answer_str.lower() == "none": # Check again after processing
                if is_math_eval_debug: print(f"True answer string became empty/None after processing: '{true_answer_str}'")
                return False
            if not predicted_answer_str or predicted_answer_str.lower() == "none": # Check again after processing
                if is_math_eval_debug: print(f"Predicted answer string became empty/None after processing: '{predicted_answer_str}'")
                return False

            true_expr = parse_latex(true_answer_str)
            if is_math_eval_debug: print(f"Sympy parsed true_expr: {true_expr}")
            pred_expr = parse_latex(predicted_answer_str)
            if is_math_eval_debug: print(f"Sympy parsed pred_expr: {pred_expr}")

            # Attempt numeric comparison first with tolerance
            try:
                if (true_expr.is_Float or true_expr.is_Integer or true_expr.is_Rational) and \
                   (pred_expr.is_Float or pred_expr.is_Integer or pred_expr.is_Rational):                 
                    if is_math_eval_debug: print(f"Comparing numerically: N(true_expr)={N(true_expr)}, N(pred_expr)={N(pred_expr)}")
                    if math.isclose(N(true_expr), N(pred_expr), rel_tol=1e-4, abs_tol=1e-9): # Adjusted tolerance slightly
                        if is_math_eval_debug: print("Numeric match FOUND.")
                        return True
                    else:
                        if is_math_eval_debug: print("Numeric match FAILED.")
            except AttributeError: # is_Float etc. might not exist if parsing failed to produce a number type
                if is_math_eval_debug: print("AttributeError during numeric check - likely not a simple number type.")
                pass # Continue to symbolic comparison

            # Symbolic comparison
            if is_math_eval_debug: print(f"Attempting true_expr.equals(pred_expr): {true_expr.equals(pred_expr)}")
            if true_expr.equals(pred_expr):
                if is_math_eval_debug: print("equals() match FOUND.")
                return True
            
            # Simplify difference
            try:
                diff = simplify(true_expr - pred_expr)
                if is_math_eval_debug: print(f"Attempting simplify(true_expr - pred_expr) == 0: simplify_diff={diff}, is zero: {diff == S.Zero}")
                if diff == S.Zero: # Compare with sympy.S.Zero for symbolic zero
                    if is_math_eval_debug: print("simplify(diff) == 0 match FOUND.")
                    return True
            except Exception as e_simplify_diff:
                if is_math_eval_debug: print(f"Error during simplify(true_expr - pred_expr): {e_simplify_diff}")

            # Compare LaTeX string of simplified expressions
            try:
                simplified_true_latex = latex(simplify(true_expr))
                simplified_pred_latex = latex(simplify(pred_expr))
                if is_math_eval_debug: print(f"Attempting latex(simplify(true)) == latex(simplify(pred)): '{simplified_true_latex}' vs '{simplified_pred_latex}'. Match: {simplified_true_latex == simplified_pred_latex}")
                if simplified_true_latex == simplified_pred_latex:
                    if is_math_eval_debug: print("Simplified LaTeX match FOUND.")
                    return True
            except Exception as e_latex_compare:
                if is_math_eval_debug: print(f"Error during simplified LaTeX comparison: {e_latex_compare}")
            
            if is_math_eval_debug: print("All SymPy comparisons FAILED.")
            return False

        except Exception as e: # Catch errors from sympy processing (parsing, simplification, etc.)
            if is_math_eval_debug: 
                # Safely get string representations for logging, even if variables are not defined due to early error
                log_true_ans = true_answer_str if 'true_answer_str' in locals() else str(dataset_item.get("true_answer"))
                log_pred_ans = predicted_answer_str if 'predicted_answer_str' in locals() else str(predicted_answer_val if 'predicted_answer_val' in locals() else result_item.get("answer"))
                print(f"SymPy Processing Error (falling back to string compare): '{log_true_ans}' vs '{log_pred_ans}'. Error: {type(e).__name__} - {e}")
            # Fall through to basic string comparison block if sympy processing fails
            pass 

    # Fallback to the most basic string comparison if sympy is not available or failed processing
    if is_math_eval_debug and not SYMPY_AVAILABLE: # Log if we are in fallback because SYMPY_AVAILABLE is False
        print(f"-- MATH DEBUG ID (String Fallback due to SYMPY_UNAVAILABLE): {result_item.get('id')} --")
    elif is_math_eval_debug: # Log if we are in fallback after a sympy processing error
        print(f"-- MATH DEBUG ID (String Fallback AFTER SymPy Error): {result_item.get('id')} --")
        
    fb_true_answer_str = str(dataset_item.get("true_answer"))
    fb_predicted_answer_val = result_item.get("answer")
    if result_item.get("prompt_type") == "self_reflection":
        fb_predicted_answer_val = result_item.get("final_answer_from_reflection", result_item.get("final_answer", fb_predicted_answer_val))
    fb_predicted_answer_str = str(fb_predicted_answer_val)

    if not fb_true_answer_str or fb_true_answer_str.lower() == "none":
        if is_math_eval_debug: print(f"Fallback: True answer string is empty/None: '{fb_true_answer_str}'")
        return False
    if not fb_predicted_answer_str or fb_predicted_answer_str.lower() == "none":
        if is_math_eval_debug: print(f"Fallback: Predicted answer string is empty/None: '{fb_predicted_answer_str}'")
        return False
        
    norm_true = fb_true_answer_str.strip().lower().replace(" ", "")
    norm_pred = fb_predicted_answer_str.strip().lower().replace(" ", "")
    latex_cmds_to_remove = ["\\frac", "\\cdot", "\\left", "\\right", "(", ")", "{", "}", "\\text", "\\mathrm", "\\%", "%"]
    for cmd in latex_cmds_to_remove:
        norm_true = norm_true.replace(cmd, "")
        norm_pred = norm_pred.replace(cmd, "")
    if is_math_eval_debug: print(f"Fallback string comparison: '{norm_true}' vs '{norm_pred}'. Match: {norm_true == norm_pred}")
    if norm_true == norm_pred:
        return True
    return False

def _is_correct_strategyqa(result_item: Dict[str, Any], dataset_item: Dict[str, Any]) -> bool:
    true_answer_bool = dataset_item.get("true_answer") # Should be a Python boolean
    
    predicted_answer_val = None
    if result_item.get("prompt_type") == "self_reflection" and "final_answer_from_reflection" in result_item:
        predicted_answer_val = result_item.get("final_answer_from_reflection")
    elif result_item.get("prompt_type") == "self_reflection" and "final_answer" in result_item:
        predicted_answer_val = result_item.get("final_answer")
    else:
        predicted_answer_val = result_item.get("answer")

    if predicted_answer_val is None or true_answer_bool is None:
        return False

    # The 'answer' field from parsing should ideally already be a boolean.
    # If it's a string like "yes", "no", it should have been converted by the parser.
    if isinstance(predicted_answer_val, bool):
        return predicted_answer_val == true_answer_bool
    
    # Fallback if predicted_answer_val is still a string (e.g. "yes", "no")
    # This indicates the parser might need adjustment or the model output was unexpected.
    if isinstance(predicted_answer_val, str):
        pred_str_lower = predicted_answer_val.lower().strip()
        if pred_str_lower in ["yes", "true"]:
            return true_answer_bool is True
        if pred_str_lower in ["no", "false"]:
            return true_answer_bool is False
            
    return False

def _is_correct_commonsenseqa(result_item: Dict[str, Any], dataset_item: Dict[str, Any]) -> bool:
    """
    Evaluates if the predicted CommonsenseQA answer (choice letter) is correct.
    (Copied from run_experiments.py)
    """
    true_answer_label = dataset_item.get("true_answer") # Ground truth label (e.g., "A")
    
    predicted_answer_val = None
    if result_item.get("prompt_type") == "self_reflection":
        if "final_answer_from_reflection" in result_item:
            predicted_answer_val = result_item.get("final_answer_from_reflection")
        elif "final_answer" in result_item:
            predicted_answer_val = result_item.get("final_answer")

    if predicted_answer_val is None:
        predicted_answer_val = result_item.get("answer")

    if predicted_answer_val is None or true_answer_label is None:
        return False

    if isinstance(predicted_answer_val, str):
        return predicted_answer_val.strip().upper() == true_answer_label.strip().upper()
    return False

# Define a dictionary for evaluation functions, similar to run_experiments.py
EVALUATION_FUNCTIONS = {
    "gsm8k": _is_correct_gsm8k_robust,
    "strategyqa": _is_correct_strategyqa,
    "math": _is_correct_math,
    "commonsenseqa": _is_correct_commonsenseqa,
}

# --- Function to load original dataset (simplified from run_experiments.py) ---
def load_original_dataset_for_eval(dataset_name: str, split: str, data_dir_base: str) -> List[Dict[str, Any]]:
    """Loads the original dataset items to get true answers and IDs."""
    # This is a simplified loader. For full robustness, you might reuse/adapt
    # the load_dataset function from run_experiments.py, but that's more complex here.
    # For now, we assume a simple structure or that the necessary info is in result files.
    # A better approach is to ensure result files contain the true_answer.
    # Let's assume for now that regenerate_summary will rely on true_answer present in the result JSONs.
    # If not, this function would need to be as robust as the one in run_experiments.py.
    # For now, returning an empty list and relying on `true_answer` in result files.
    # A proper implementation would load the actual dataset similar to run_experiments.py
    print(f"Attempting to load original dataset: {dataset_name}, split: {split} from {data_dir_base}")
    # This part needs to be implemented robustly if true_answer is not always in result files.
    # For example, using a slimmed down version of load_dataset from run_experiments.py:
    
    data_path = Path(data_dir_base) / dataset_name
    raw_data = []

    if dataset_name == "gsm8k":
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("gsm8k", "main")
        for item in ds[split]: # split is usually 'test' for gsm8k
            raw_data.append({"id": f"gsm8k_{split}_{len(raw_data)}", "question": item["question"], "true_answer": item["answer"].split("####")[-1].strip()})
    elif dataset_name == "math":
        # Simplified: Assumes MATH test files are JSONs in data/math/test/*/*.json
        # This is a basic loader and might need adjustment based on your exact MATH structure
        math_split_dir = data_path / split
        if math_split_dir.is_dir():
            for subject_dir in math_split_dir.iterdir():
                if subject_dir.is_dir():
                    for problem_file in subject_dir.glob("*.json"):
                        try:
                            with open(problem_file, 'r') as f:
                                content = json.load(f)
                                solution = content.get("solution", "")
                                boxed_match = re.search(r"\\boxed{(.+?)}", solution)
                                answer = boxed_match.group(1).strip() if boxed_match else solution.strip()
                                raw_data.append({
                                    # Consistent ID: use problem_file.name to include .json
                                    "id": f"math_{split}_{problem_file.name}", 
                                    "question": content.get("problem", ""), 
                                    "true_answer": answer
                                })
                        except Exception as e:
                            print(f"Error loading MATH problem {problem_file}: {e}")
        else:
            print(f"MATH dataset directory not found: {math_split_dir}")
    elif dataset_name == "strategyqa":
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("ChilleD/StrategyQA") # Using the one that works
        for item in ds[split]: # e.g. split = "test"
            raw_data.append({"id": f"strategyqa_{split}_{len(raw_data)}", "question": item["question"], "true_answer": item["answer"]})
    elif dataset_name == "commonsenseqa":
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("commonsense_qa")
        # CommonsenseQA HF dataset uses 'validation' for dev split.
        # For 'test' split, which often has no public labels, we should also use 'validation' for evaluation.
        hf_split = "validation" if split == "dev" or split == "test" else split
        print(f"INFO: For CommonsenseQA, loading HuggingFace split: '{hf_split}' (original request was for '{split}')")
        for item in ds[hf_split]:
            question_text = item["question"]
            choices_text = "\nChoices:\n" + "\n".join([f"({label}) {text}" for label, text in zip(item["choices"]["label"], item["choices"]["text"])])
            raw_data.append({
                "id": f"csqa_{split}_{len(raw_data)}", 
                "question": question_text + choices_text, 
                "true_answer": item["answerKey"].upper()
            })
    else:
        print(f"Dataset loader for {dataset_name} not fully implemented in regenerate_summary.py. True answers must be in result files.")

    return raw_data


# --- Main Loading and Evaluation Logic ---
def load_and_evaluate_results(model_name: str, dataset_name: str, dataset_split: str, data_dir_base: str, force_reparse: bool):
    results_dir_base_original = Path("results") / dataset_name / model_name
    results_dir_base_alt = Path("results") / dataset_name # For strategyqa

    # For StrategyQA, results files might be directly under results/strategyqa and named like <model_name>_<prompt_type>_...
    # The script normally expects them under results/strategyqa/<model_name> and named <prompt_type>_...
    # We will adjust which directory to scan and how to identify prompt types for strategyqa.
    
    current_results_path_to_scan = results_dir_base_original
    filename_prefix_is_model_specific = False

    if dataset_name in ["strategyqa", "commonsenseqa"]:
        # Check if the model-specific directory is empty or has few relevant JSONs
        # This is a heuristic. A more robust way might be to always check the alternative path.
        try:
            original_path_files = list(results_dir_base_original.glob("*.json"))
            # If gpt-4/ (or other model) dir exists but only has a sparse summary (like the one previously generated)
            # or no files matching prompt types, then prefer the alternative path.
            if not any(f.name.split('_')[0] in PROMPT_TEMPLATE_CLASSES.keys() for f in original_path_files if '_' in f.name) and results_dir_base_alt.exists():
                print(f"INFO: For StrategyQA and model {model_name}, model-specific dir {results_dir_base_original} seems sparse or lacks direct prompt files. Checking {results_dir_base_alt} instead.")
                current_results_path_to_scan = results_dir_base_alt
                filename_prefix_is_model_specific = True # Files here are like <model>_<prompt>_...
            elif not results_dir_base_original.exists() and results_dir_base_alt.exists():
                print(f"INFO: For StrategyQA and model {model_name}, model-specific dir {results_dir_base_original} does not exist. Checking {results_dir_base_alt}.")
                current_results_path_to_scan = results_dir_base_alt
                filename_prefix_is_model_specific = True
            else:
                print(f"INFO: For StrategyQA and model {model_name}, using standard path: {results_dir_base_original}")
        except Exception as e:
            print(f"Warning: Error while checking paths for StrategyQA: {e}. Defaulting to original path.")

    all_loaded_results: List[Dict[str, Any]] = []

    print(f"Loading results for model '{model_name}' on dataset '{dataset_name}' from {current_results_path_to_scan}")

    # Load original dataset items for true answers (if needed for re-parsing or robust eval)
    original_dataset_items = load_original_dataset_for_eval(dataset_name, dataset_split, data_dir_base)
    original_dataset_items_map = {item["id"]: item for item in original_dataset_items}
    if not original_dataset_items_map:
        print("Warning: Could not load original dataset items. Evaluation might be affected if true answers are not in result files or re-parsing is needed.")

    prompt_types_present_in_results = set()
    
    # Debug: Print the keys it will use for matching prefixes
    print(f"DEBUG: PROMPT_TEMPLATE_CLASSES keys available: {list(PROMPT_TEMPLATE_CLASSES.keys())}")
    print(f"DEBUG: Scanning for JSON files in: {current_results_path_to_scan}")

    # First, discover which prompt types have result files based on filename prefixes
    files_to_check_for_discovery = list(current_results_path_to_scan.glob("*.json"))
    if dataset_name in ["strategyqa", "commonsenseqa"]:
        files_to_check_for_discovery.extend(list(current_results_path_to_scan.glob("*.jsonl")))
        files_to_check_for_discovery = sorted(list(set(files_to_check_for_discovery)))

    for f_path in files_to_check_for_discovery: # Check all JSON files in the model_name dir
        print(f"DEBUG: Checking file: {f_path.name}") # Debug: print each file being checked
        for pt_class_name in PROMPT_TEMPLATE_CLASSES.keys():
            expected_prefix = pt_class_name + "_"
            if filename_prefix_is_model_specific:
                # For strategyqa, files are <model_name>_<prompt_type_name>_....json or .jsonl
                # So, we check if file starts with <model_name>_<pt_class_name>_
                model_specific_prefix = model_name + "_" + pt_class_name + "_"
                if f_path.name.startswith(model_specific_prefix):
                    prompt_types_present_in_results.add(pt_class_name)
                    print(f"DEBUG: Matched {f_path.name} to prompt type {pt_class_name} (model-specific prefix)")
                    break 
            elif f_path.name.startswith(expected_prefix):
                prompt_types_present_in_results.add(pt_class_name)
                print(f"DEBUG: Matched {f_path.name} to prompt type {pt_class_name}") # Debug: print match
                break
    
    print(f"Found result files for prompt types: {list(prompt_types_present_in_results)}")

    for prompt_type_name in prompt_types_present_in_results:
        print(f"  Loading results for prompt type: {prompt_type_name}")

        file_pattern = ""
        if filename_prefix_is_model_specific:
            file_pattern = f"{model_name}_{prompt_type_name}_*.json"
        else:
            file_pattern = f"{prompt_type_name}_*.json"
        
        # Handle .jsonl as well, as some strategyqa files are .jsonl
        candidate_files = list(current_results_path_to_scan.glob(file_pattern))
        if dataset_name in ["strategyqa", "commonsenseqa"]:
            if filename_prefix_is_model_specific:
                candidate_files.extend(list(current_results_path_to_scan.glob(f"{model_name}_{prompt_type_name}_*.jsonl")))
            else:
                candidate_files.extend(list(current_results_path_to_scan.glob(f"{prompt_type_name}_*.jsonl")))
            # Deduplicate if needed, though glob shouldn't produce pure duplicates unless filenames are identical with different extensions
            candidate_files = sorted(list(set(candidate_files))) 


        if not candidate_files:
            print(f"    No result files found for prompt type '{prompt_type_name}' with pattern '{file_pattern}'.")
            continue

        largest_file = None
        max_size = -1
        for f_path in candidate_files:
            try:
                size = f_path.stat().st_size
                if size > max_size:
                    max_size = size
                    largest_file = f_path
            except OSError:
                print(f"    Warning: Could not stat file {f_path}. It might have been deleted.")
                pass 

        if not largest_file:
            print(f"    Could not determine the primary result file for '{prompt_type_name}' from candidates: {candidate_files}")
            continue
            
        print(f"    Selected primary result file: {largest_file.name} (size: {max_size} bytes)")
        files_to_load = [largest_file] 

        prompt_results_for_type: List[Dict[str, Any]] = []
        for file_path in files_to_load: # This loop will now run once for the largest file
            try:
                with open(file_path, 'r') as f:
                    if file_path.name.endswith(".jsonl"):
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    content_item = json.loads(line)
                                    prompt_results_for_type.append(content_item)
                                except json.JSONDecodeError as e_item:
                                    print(f"    Warning: Could not decode JSON line in {file_path}: {line[:100]}... Error: {e_item}")
                    elif file_path.name.endswith(".json"):
                        content = json.load(f)
                        if isinstance(content, list):
                            prompt_results_for_type.extend(content)
                        elif isinstance(content, dict):
                            print(f"    Warning: Loaded a single dictionary from {file_path}, expected a list of results. Wrapping it.")
                            prompt_results_for_type.append(content)
                    else:
                        print(f"    Warning: Skipping file with unknown extension: {file_path}")
            except json.JSONDecodeError: # This might catch general issues if json.load() was still called incorrectly
                print(f"    Warning: Could not decode JSON from {file_path}")
            except Exception as e:
                print(f"    Warning: Error loading {file_path}: {e}")

        if not prompt_results_for_type:
            print(f"    No results loaded for prompt type '{prompt_type_name}' after selecting file.")
            continue

        # Re-parse if forced
        if force_reparse:
            print(f"    Re-parsing {len(prompt_results_for_type)} results for '{prompt_type_name}'...")
            try:
                # Get the appropriate prompt template for parsing
                # dataset_name should be consistent for all results of this type
                current_dataset_name = prompt_results_for_type[0].get("dataset_name", dataset_name)
                prompt_template_instance = get_prompt_template(prompt_type_name, dataset_name=current_dataset_name)
                
                reparsed_results = []
                for res_item in prompt_results_for_type:
                    raw_output = res_item.get("raw_output")
                    if raw_output is None:
                        # If raw_output is missing, try model_output_full as a fallback, common in older files
                        raw_output = res_item.get("model_output_full")
                    
                    if raw_output is not None:
                        question_text = res_item.get("question", "") # For prompts that might use it
                        if prompt_type_name == "self_consistency":
                            # Self-consistency output is a list of raw strings from samples
                            parsed_details = prompt_template_instance.parse_output(raw_output, question=question_text)
                        elif prompt_type_name == "self_reflection":
                            # Self-reflection parse_output expects initial and reflected outputs
                            # This part is tricky if only raw_output (final) is stored.
                            # For simplicity, we assume raw_output is the final one for re-parsing SR.
                            # A more robust re-parse would need initial_solution & final_reflection_output fields stored.
                            # Let's assume `raw_output` for SR in JSON is the final reflected one.
                            # And the parse_output for SR can handle this (might need adjustment in prompts.py)
                            # For now, passing raw_output as both if only one is available from old files.
                            initial_out = res_item.get("initial_reasoning", raw_output) # Fallback for older files
                            final_out = raw_output
                            parsed_details = prompt_template_instance.parse_output(initial_model_output=initial_out, reflection_model_output=final_out, question=question_text)
                        else:
                            parsed_details = prompt_template_instance.parse_output(raw_output, question=question_text)
                        
                        # Update the item with re-parsed details
                        # Ensure we keep essential original fields like id, true_answer etc.
                        updated_item = {**res_item, **parsed_details} # Merge, parsed_details might overwrite 'answer'
                        reparsed_results.append(updated_item)
                    else:
                        print(f"    Warning: Missing 'raw_output' or 'model_output_full' for re-parsing item {res_item.get('id')}. Skipping re-parse for this item.")
                        reparsed_results.append(res_item) # Keep original if no raw output
                all_loaded_results.extend(reparsed_results)
            except ValueError as e:
                print(f"    Error getting prompt template for '{prompt_type_name}' (dataset: {dataset_name}): {e}. Skipping re-parsing for this type.")
                all_loaded_results.extend(prompt_results_for_type) # Add original if re-parsing fails
            except Exception as e:
                print(f"    Unexpected error during re-parsing for '{prompt_type_name}': {e}. Skipping re-parsing for this type.")
                all_loaded_results.extend(prompt_results_for_type) # Add original if re-parsing fails
        else:
            all_loaded_results.extend(prompt_results_for_type)
    
    if not all_loaded_results:
        print("No results loaded across all prompt types. Cannot generate summary.")
        return

    # Get the evaluation function for the dataset
    if dataset_name not in EVALUATION_FUNCTIONS:
        print(f"Error: No evaluation function defined for dataset '{dataset_name}'. Cannot generate summary.")
        return
    evaluation_function = EVALUATION_FUNCTIONS[dataset_name]

    evaluate_and_print_summary(all_loaded_results, model_name, dataset_name, evaluation_function, original_dataset_items_map)


# --- Evaluation Functions (adapted from run_experiments.py) ---

def calculate_accuracy_with_ci(
    results_for_type: List[Dict[str, Any]], 
    original_dataset_items_map: Dict[str, Dict[str, Any]], # Map item_id to original dataset item
    evaluation_function: Callable, 
    confidence_level: float = 0.95
) -> Tuple[float, float, float, int, int]: # Return correct_count, total_evaluable
    correct_predictions = 0
    total_evaluable_predictions = 0

    for res_item in results_for_type:
        original_item = original_dataset_items_map.get(res_item["id"])
        if not original_item:
            # This case should ideally be handled before calling this function,
            # by ensuring results_for_type only contains items with corresponding original_dataset_items.
            # print(f"Warning: Original dataset item not found for ID {res_item['id']} in calculate_accuracy_with_ci. Skipping.")
            continue

        # Check if the item is evaluable (has an answer field from parsing)
        has_answer_field = res_item.get("answer") is not None
        has_final_answer_field_sr = res_item.get("prompt_type") == "self_reflection" and \
                                    (res_item.get("final_answer_from_reflection") is not None or \
                                     res_item.get("final_answer") is not None) # check older key too

        if has_answer_field or has_final_answer_field_sr:
            total_evaluable_predictions += 1
            # Pass the debug_math_id to the evaluation_function if it's _is_correct_math
            current_id_for_debug = None
            # This is a bit of a hack to get args.debug_math_id down here
            # A cleaner way would be to pass it through the call stack or make it globally accessible if safe
            # For now, assuming it might be accessible via some shared context or if we modify main to pass it.
            # This specific edit won't work directly without changing how debug_math_id is passed to calculate_accuracy_with_ci
            # For the purpose of this call, we assume debug_math_id is available in this scope if needed by _is_correct_math
            # The user will enable it manually in _is_correct_math for now.
            if evaluation_function(res_item, original_item): # If eval func is _is_correct_math, it uses its internal flag
                correct_predictions += 1
    
    if total_evaluable_predictions == 0:
        return 0.0, 0.0, 0.0, 0, 0

    accuracy = correct_predictions / total_evaluable_predictions
    
    # Wilson score interval for binomial proportion
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    n = total_evaluable_predictions
    p = accuracy # Use the calculated accuracy 
    
    denominator = 1 + z**2 / n
    term1_num = p + z**2 / (2 * n)
    term2_num_sqrt = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    
    ci_lower = (term1_num - term2_num_sqrt) / denominator
    ci_upper = (term1_num + term2_num_sqrt) / denominator
    
    return accuracy, max(0.0, ci_lower), min(1.0, ci_upper), correct_predictions, total_evaluable_predictions


def evaluate_and_print_summary(
    all_model_results: List[Dict[str, Any]], model_name: str, dataset_name: str, evaluation_function: Callable, original_dataset_items_map: Dict[str, Dict[str, Any]]
):
    """
    Evaluates and prints the summary of results, similar to run_experiments.py.
    """
    # Collect every line we print so we can also dump them to a file
    summary_lines: List[str] = []

    def _out(line: str):
        summary_lines.append(line)
        print(line)

    _out("\n=== Evaluation Results ===")
    _out(f"Model: {model_name}, Dataset: {dataset_name}")
    _out(f"Total examples loaded: {len(all_model_results)}")

    if not all_model_results:
        _out("No results to evaluate.")
        return

    # Filter results to only include those that have a corresponding original item
    # This is important if some result IDs don't match any loaded original dataset item.
    valid_results_with_original = []
    for res_item in all_model_results:
        if res_item["id"] in original_dataset_items_map:
            valid_results_with_original.append(res_item)
        else:
            _out(f"  Warning: Result item with ID '{res_item['id']}' not found in original dataset map. Skipping for overall summary.")
    
    _out(f"Total examples loaded and matched with original dataset: {len(valid_results_with_original)}")

    if not valid_results_with_original:
        _out("No results to evaluate after matching with original dataset.")
        # ... (save summary_lines to file logic) ...
        return

    # Calculate overall accuracy using the specific evaluation function
    # calculate_accuracy_with_ci now returns correct_count and total_evaluable
    overall_accuracy, overall_ci_lower, overall_ci_upper, overall_correct, overall_evaluable = calculate_accuracy_with_ci(
        results_for_type=valid_results_with_original, 
        original_dataset_items_map=original_dataset_items_map, 
        evaluation_function=evaluation_function
    )
    
    if overall_evaluable > 0:
        summary_lines.append(f"Overall accuracy: {overall_accuracy:.4f} ({overall_correct}/{overall_evaluable} valid examples)")
        summary_lines.append(f"95% CI: [{overall_ci_lower:.4f}, {overall_ci_upper:.4f}]")
    else:
        summary_lines.append(f"Overall accuracy: No evaluable examples found.")
    summary_lines.append("")

    # Use PROMPT_TEMPLATE_CLASSES.keys() or a predefined list
    prompt_types_to_summarize = list(PROMPT_TEMPLATE_CLASSES.keys())

    for prompt_type in prompt_types_to_summarize:
        # Filter results for the current prompt type AND ensure they have a match in original_dataset_items_map
        type_results_all_matched = [
            res for res in valid_results_with_original if res.get("prompt_type") == prompt_type
        ]
            
        if not type_results_all_matched:
            # _out(f"{prompt_type} accuracy: No data for this type or no match with original dataset.")
            continue # Skip if no data for this type

        acc, ci_low, ci_high, type_correct, type_evaluable = calculate_accuracy_with_ci(
            results_for_type=type_results_all_matched, 
            original_dataset_items_map=original_dataset_items_map, 
            evaluation_function=evaluation_function
        )
        
        if type_evaluable > 0:
            summary_lines.append(f"{prompt_type} accuracy: {acc:.4f} ({type_correct}/{type_evaluable})")
            summary_lines.append(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        else:
            summary_lines.append(f"{prompt_type} accuracy: No evaluable items")
        summary_lines.append("")

    # ----------  write summary to disk  ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"results/{dataset_name}/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"summary_{ts}.txt"
    out_file.write_text("\n".join(summary_lines))
    _out(f"\nSummary written to {out_file}")

    # --- Call plot_error_distribution --- #
    if valid_results_with_original: 
        results_for_error_plot = []
        for res_item in valid_results_with_original:
            original_item = original_dataset_items_map.get(res_item["id"])
            if original_item:
                item_for_plot = res_item.copy()
                item_for_plot["true_answer"] = original_item.get("true_answer")
                if item_for_plot.get("parsing_error") and not (item_for_plot.get("answer") == item_for_plot.get("true_answer")):
                    item_for_plot["error_type"] = "Parsing Error/No Answer" 
                results_for_error_plot.append(item_for_plot)
            
        if results_for_error_plot:
            print(f"\nGenerating error distribution plot for {model_name} on {dataset_name}...")
            plot_error_distribution(results_for_error_plot, model_name, dataset_name)
            print(f"Error distribution plot saved to plots/{model_name}_{dataset_name}_error_distribution.png")
        else:
            print("No valid items with true_answers found for error plotting.")

# --- Main part of the script ---
def main():
    parser = argparse.ArgumentParser(description="Regenerate summary from experiment results.")
    parser.add_argument("--model_name", type=str, default="gpt-4", help="Name of the model to evaluate.")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="gsm8k", 
        choices=list(EVALUATION_FUNCTIONS.keys()), # Use keys from eval functions dict
        help="Name of the dataset."
    )
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split used for original experiments (e.g., test, dev). Needed to load true answers.")
    parser.add_argument("--data_dir_base", type=str, default="data", help="Base directory where original dataset subdirectories are located.")
    parser.add_argument("--force_reparse", action="store_true", help="Force re-parsing of raw_output fields.")
    parser.add_argument("--debug_math_id", type=str, default=None, help="Set a specific MATH item ID for debug printing in _is_correct_math.")
    args = parser.parse_args()
    
    # Modify _is_correct_math directly if args.debug_math_id is set (this is a bit of a hack for demonstration)
    # A cleaner way would be to pass this ID down through function calls or use a global/context variable.
    # For now, we are asking user to manually edit the flag in the function based on this arg if they want it.
    if args.debug_math_id:
        print(f"INFO: Debugging requested for MATH ID: {args.debug_math_id}. Ensure is_math_eval_debug is set for this ID in _is_correct_math.")

    load_and_evaluate_results(args.model_name, args.dataset_name, args.dataset_split, args.data_dir_base, args.force_reparse)

if __name__ == "__main__":
    # Ensure your Python environment has numpy and scipy installed:
    # pip install numpy scipy
    main() 