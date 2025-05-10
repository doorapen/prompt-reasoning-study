import json
import os
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.stats import norm
import re
from src.prompts import (
    PROMPT_TEMPLATES,
    get_prompt_template,
    SelfReflectionPrompt,
    PromptTemplate,
    SelfConsistencyPrompt,
)
from datetime import datetime

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

def _is_correct_gsm8k_robust(result_item: Dict[str, Any]) -> bool:
    true_answer_str = result_item.get("true_answer")
    
    predicted_answer_str = None
    # In run_experiments.py, for self_reflection, 'final_answer' is copied to 'answer'
    # before saving. However, _is_correct_gsm8k_robust itself also checks final_answer
    # first, which is good for robustness if used independently or if JSONs are older.
    if result_item.get("prompt_type") == "self_reflection" and "final_answer" in result_item:
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

# --- Evaluation Functions (adapted from run_experiments.py) ---

def calculate_accuracy_with_ci(results: List[Dict[str, Any]], confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculates accuracy and confidence interval for a list of results.
    Uses _is_correct_gsm8k_robust for determining correctness.
    The 'results' list is expected to be pre-filtered for evaluable items.
    """
    correct_predictions = 0
    
    # total_predictions will be the count of items in the 'results' list,
    # as this list is already filtered for evaluable items before being passed here.
    total_predictions = len(results)

    if total_predictions == 0:
        return 0.0, 0.0, 0.0

    for res in results:
        # Determine correctness using the logic from run_experiments.py
        if _is_correct_gsm8k_robust(res):
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    
    # Wilson score interval for binomial proportion
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    n = total_predictions
    p = accuracy # Use the calculated accuracy 
    
    denominator = 1 + z**2 / n
    term1_num = p + z**2 / (2 * n)
    term2_num_sqrt = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    
    ci_lower = (term1_num - term2_num_sqrt) / denominator
    ci_upper = (term1_num + term2_num_sqrt) / denominator
    
    return accuracy, max(0.0, ci_lower), min(1.0, ci_upper)


def evaluate_and_print_summary(
    all_model_results: List[Dict[str, Any]], model_name: str, dataset_name: str
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

    # Calculate overall accuracy (without CI, matching user's deepseek output format)
    overall_correct = 0
    overall_total_valid_for_accuracy_calc = 0
    
    # Filter results that can be used for accuracy calculation
    # These are items that are not errors and have the necessary answer fields.
    # The definition of "evaluable" here should align with run_experiments.py's evaluate_gsm8k_main
    
    evaluable_results_for_overall = []
    for res_overall in all_model_results:
        # In run_experiments.py, evaluate_gsm8k_main filters out items with r.get("error")
        if res_overall.get("error"):
            continue
        # _is_correct_gsm8k_robust handles missing true_answer or predicted_answer internally
        evaluable_results_for_overall.append(res_overall)

    overall_total_valid_for_accuracy_calc = len(evaluable_results_for_overall)

    for res_evaluable in evaluable_results_for_overall:
        if _is_correct_gsm8k_robust(res_evaluable):
            overall_correct += 1
            
    # The old way of calculating overall_correct and overall_total_valid_for_accuracy_calc
    # is now replaced by the loop above using _is_correct_gsm8k_robust

    if overall_total_valid_for_accuracy_calc > 0:
        overall_accuracy_val = overall_correct / overall_total_valid_for_accuracy_calc
        _out(f"Overall accuracy: {overall_accuracy_val:.4f} (calculated over {overall_total_valid_for_accuracy_calc} valid examples)")
    else:
        _out("Overall accuracy: N/A (no valid examples for calculation)")

    # Group results by prompt_type
    results_by_prompt: Dict[str, List[Dict[str, Any]]] = {}
    for res in all_model_results:
        # Filter out items with errors, similar to evaluate_gsm8k_main in run_experiments.py
        if res.get("error"):
            continue
        
        prompt_type = res.get("prompt_type")
        if prompt_type:
            if prompt_type not in results_by_prompt:
                results_by_prompt[prompt_type] = []
            results_by_prompt[prompt_type].append(res)

    # Use the prompt order from PROMPT_TEMPLATES for consistent output
    sorted_prompt_types = [name for name in PROMPT_TEMPLATES.keys() if name in results_by_prompt]
    # Add any other found prompt types not in PROMPT_TEMPLATES (e.g. "unknown")
    for pt in results_by_prompt.keys():
        if pt not in sorted_prompt_types:
            sorted_prompt_types.append(pt)


    for prompt_type in sorted_prompt_types:
        prompt_results = results_by_prompt[prompt_type]
        
        # Filter for valid results for this specific prompt type for CI calculation
        valid_prompt_results_for_ci = [
            pr for pr in prompt_results 
            if not pr.get("error") and "true_answer" in pr and "answer" in pr
        ]
        
        if not valid_prompt_results_for_ci:
            _out(f"\n{prompt_type} accuracy: 0.0000 (No valid results for CI calculation)")
            _out(f"95% CI: [0.0000, 0.0000]")
            continue

        accuracy, ci_lower, ci_upper = calculate_accuracy_with_ci(valid_prompt_results_for_ci)
        _out(f"\n{prompt_type} accuracy: {accuracy:.4f}")
        _out(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # ----------  write summary to disk  ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"results/{dataset_name}/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"summary_{ts}.txt"
    out_file.write_text("\n".join(summary_lines))
    _out(f"\nSummary written to {out_file}")

# --- Main part of the script ---
def main():
    # model_to_evaluate = "gpt-4"  # Change this if you want to evaluate another model
    # model_to_evaluate = "deepseek-r1"
    model_to_evaluate = "gpt-4"
    dataset_name = "gsm8k"
    
    # --- CONTROL RE-PARSING ---
    # Set to False to trust 'answer' fields already in JSON files (for reproducing old summaries).
    # Set to True to re-parse all 'raw_output' fields using the current parsing logic.
    force_reparse = True
    # ---

    base_results_dir = Path(f"results/{dataset_name}/{model_to_evaluate}")
    if not base_results_dir.is_dir():
        print(f"Error: Base results directory not found: {base_results_dir}")
        return

    prompt_names_to_check = list(PROMPT_TEMPLATES.keys()) 
    all_results_for_model: List[Dict[str, Any]] = []

    print(f"Looking for results in: {base_results_dir}")
    print(f"Checking for prompt types: {prompt_names_to_check}")

    for p_name in prompt_names_to_check:
        prompt_dir = base_results_dir / p_name
        if not prompt_dir.is_dir():
            # This is normal if not all prompt types were run for the model
            # print(f"Info: Directory not found for prompt type {p_name} at {prompt_dir}, skipping.")
            continue

        # Attempt to load the main summary file first
        # Result files are named like <prompt_name>_<timestamp>.json
        # (as per save_results in run_experiments.py)
        summary_files = sorted(list(prompt_dir.glob(f"{p_name}_*.json"))) # Sort to get latest if multiple
        
        loaded_from_summary = False
        if summary_files:
            latest_summary_file = summary_files[-1] 
            print(f"Loading results for prompt '{p_name}' from summary file: {latest_summary_file}")
            try:
                with open(latest_summary_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data: # Ensure prompt_type is set
                            if "prompt_type" not in item:
                                item["prompt_type"] = p_name 
                        all_results_for_model.extend(data)
                        loaded_from_summary = True
                    else:
                        print(f"Warning: Expected a list in summary file {latest_summary_file}, got {type(data)}.")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from summary file {latest_summary_file}.")
            except Exception as e:
                print(f"Error: Could not load summary file {latest_summary_file}: {e}.")
        
        if not loaded_from_summary:
            # Fallback: If no summary file was loaded, try to load from batch files
            batch_files = sorted(list(prompt_dir.glob("batch_*.json")))
            if batch_files:
                print(f"No summary file found for '{p_name}'. Attempting to load from {len(batch_files)} batch files in {prompt_dir}...")
                prompt_specific_batch_results = []
                for batch_file in batch_files:
                    # print(f"  Loading from batch file: {batch_file.name}") # Optional: more verbose logging
                    try:
                        with open(batch_file, 'r') as f:
                            batch_data = json.load(f)
                            if isinstance(batch_data, list):
                                for item in batch_data: # Ensure prompt_type is set
                                    if "prompt_type" not in item:
                                        item["prompt_type"] = p_name
                                prompt_specific_batch_results.extend(batch_data)
                            else:
                                print(f"Warning: Expected a list in batch file {batch_file}, got {type(batch_data)}. Skipping.")
                    except json.JSONDecodeError:
                        print(f"Error: Could not decode JSON from batch file {batch_file}. Skipping.")
                    except Exception as e:
                        print(f"Error: Could not load batch file {batch_file}: {e}. Skipping.")
                
                if prompt_specific_batch_results:
                    all_results_for_model.extend(prompt_specific_batch_results)
                    print(f"  Successfully loaded {len(prompt_specific_batch_results)} items from batch files for '{p_name}'.")
                else:
                    print(f"  No data loaded from batch files for '{p_name}'.")
            # else: # No summary and no batch files
                # print(f"Info: No summary or batch files found for prompt type '{p_name}' in {prompt_dir}.")


    if not all_results_for_model:
        print(f"No results loaded for model '{model_to_evaluate}'. Cannot generate summary.")
        return

    # --- Re-parse the 'answer' field using current prompt logic ---
    print("\nRe-parsing loaded results to ensure 'answer' field is correctly formatted...")
    reparsed_results = []
    for item_index, item_data in enumerate(all_results_for_model): # Use enumerate for potential logging
        current_item = dict(item_data) # Work on a copy
        prompt_type_from_item = current_item.get("prompt_type")

        if not prompt_type_from_item:
            print(f"Warning: Item missing 'prompt_type', cannot re-parse. Item ID: {current_item.get('id')}")
            reparsed_results.append(current_item)
            continue

        # --- Stage 1: Attempt to get parsed_details IF raw data is available ---
        attempted_parse_details = None # Holds the result of a parsing attempt if made
        raw_data_was_available_for_parse = False

        try:
            template = get_prompt_template(prompt_type_from_item)
            
            if isinstance(template, SelfReflectionPrompt):
                initial_out = current_item.get("initial_output", current_item.get("initial_reasoning"))
                reflection_out = current_item.get("reflection_output", current_item.get("reflection"))
                if initial_out is not None and reflection_out is not None:
                    raw_data_was_available_for_parse = True
                    attempted_parse_details = template.parse_output(initial_out, reflection_out)
                else: 
                    raw_out_fallback = current_item.get("raw_output", current_item.get("model_output_full"))
                    if raw_out_fallback is not None:
                        raw_data_was_available_for_parse = True
                        if current_item.get("prompt_type") == "self_reflection": # Log only if actually SR
                            print(f"Info: SelfReflection item {current_item.get('id')} using fallback parsing due to missing specific fields.")
                        base_parser = PromptTemplate(f"sr_fallback_parser_for_{current_item.get('id')}")
                        attempted_parse_details = base_parser.parse_output(raw_out_fallback)
                    else:
                        # No data for SR parsing
                        if current_item.get("prompt_type") == "self_reflection":
                             print(f"Info: SelfReflection item {current_item.get('id')} missing specific fields AND raw_output/model_output_full. Cannot re-parse.")
                        # attempted_parse_details remains None, raw_data_was_available_for_parse is False
            
            elif isinstance(template, SelfConsistencyPrompt):
                sample_outputs = current_item.get("sample_outputs")
                if sample_outputs and isinstance(sample_outputs, list):
                    raw_data_was_available_for_parse = True
                    attempted_parse_details = template.parse_output(sample_outputs)
                else:
                    print(f"Warning: SelfConsistency item {current_item.get('id')} missing 'sample_outputs' or not a list. Cannot re-parse.")
                    # attempted_parse_details remains None, raw_data_was_available_for_parse is False

            else: # ZeroShot, FewShot, CoT, ReAct
                full_output = current_item.get("raw_output", current_item.get("model_output_full"))
                if full_output is not None:
                    raw_data_was_available_for_parse = True
                    attempted_parse_details = template.parse_output(full_output)
                else:
                    # This case is where the "Warning: No 'raw_output' or 'model_output_full' found..." is triggered
                    print(f"Warning: No 'raw_output' or 'model_output_full' found for item ID: {current_item.get('id')}, prompt_type: {prompt_type_from_item}. Cannot re-parse.")
                    # attempted_parse_details remains None, raw_data_was_available_for_parse is False
        
        except ValueError as e: 
            print(f"Warning: Could not get prompt template for type '{prompt_type_from_item}'. Error: {e}. Item ID: {current_item.get('id')}")
            # attempted_parse_details remains None, raw_data_was_available_for_parse is False
        except Exception as e: 
            print(f"Error during parsing attempt for item ID {current_item.get('id')} (type '{prompt_type_from_item}'): {e}")
            # import traceback; print(traceback.format_exc()) # For debugging
            # attempted_parse_details might have parsing_error, but we treat it as a failed attempt to get a clean answer
            # If an exception occurs during template.parse_output(), attempted_parse_details might be partially formed or None.
            # For safety, if an exception occurs here, we consider the re-parse attempt as not providing a usable new answer.
            # raw_data_was_available_for_parse might be true if data was there but parsing failed.
            # Let's ensure attempted_parse_details reflects a parsing failure if an exception happened.
            if attempted_parse_details is None: # If not already set by a more specific error
                 attempted_parse_details = {"answer": None, "parsing_error": f"Parsing attempt failed: {str(e)}"}


        # --- Stage 2: Decide whether to update the item's answer ---
        original_answer_is_bad_or_missing = (
            current_item.get("answer") is None or
            str(current_item.get("answer")).strip() == "" or
            str(current_item.get("answer")).strip().startswith("ERROR")
        )

        should_update_from_this_parse_attempt = False
        if force_reparse:
            # If forcing reparse, we update *only if* we had raw data AND a parse attempt was made (details not None).
            # This means if raw data was missing, we DO NOT update, preserving the original answer.
            if raw_data_was_available_for_parse and attempted_parse_details is not None:
                should_update_from_this_parse_attempt = True
        else: # Not force_reparse (force_reparse = False)
            # If not forcing reparse, we update *only if* the original answer was bad
            # AND we had raw data to attempt a new parse AND the parse attempt was made.
            if original_answer_is_bad_or_missing:
                if raw_data_was_available_for_parse and attempted_parse_details is not None:
                    should_update_from_this_parse_attempt = True
        
        if should_update_from_this_parse_attempt:
            current_item["answer"] = attempted_parse_details.get("answer")
            
            # Update parsing_error field
            if "parsing_error" in attempted_parse_details and current_item["answer"] is None:
                current_item["parsing_error"] = attempted_parse_details["parsing_error"]
            elif "parsing_error" in current_item and current_item["answer"] is not None:
                # Clear old parsing error if a new answer is successfully parsed
                del current_item["parsing_error"]

            # Update final_answer for self_reflection
            if prompt_type_from_item == "self_reflection":
                if "final_answer_from_reflection" in attempted_parse_details:
                    current_item["final_answer"] = attempted_parse_details.get("final_answer_from_reflection")
                elif attempted_parse_details.get("answer") is not None: 
                    # Fallback to the main parsed answer if specific key is missing but answer exists
                    current_item["final_answer"] = attempted_parse_details.get("answer")
                # If attempted_parse_details["answer"] is None, and original final_answer existed,
                # this logic will update final_answer to None if the new 'answer' is None.
                # This is generally correct if we are committing to the re-parsed result.
        # Else (should_update_from_this_parse_attempt is False):
        # The item's original "answer", "final_answer", and "parsing_error" are preserved.
        
        reparsed_results.append(current_item)
            
    all_results_for_model = reparsed_results # Use the re-parsed results for evaluation
    # --- End of Re-parsing ---

    evaluate_and_print_summary(all_results_for_model, model_to_evaluate, dataset_name)

if __name__ == "__main__":
    # Ensure your Python environment has numpy and scipy installed:
    # pip install numpy scipy
    main() 