import json
import os
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import re
import argparse 
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 

# --- SymPy Import (TOP LEVEL) ---  #
SYMPY_AVAILABLE = False
try:
    from sympy import sympify, simplify, N, latex, S
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
    print("INFO: generate_combined_error_plots.py: SymPy imported successfully at top level.")
except ImportError as e:
    print(f"WARNING: generate_combined_error_plots.py: SymPy Import Error at top level: {e}. MATH eval will use string fallback.")
except Exception as e:
    print(f"WARNING: generate_combined_error_plots.py: Other error during SymPy import at top level: {e}. MATH eval will use string fallback.")
# --- End SymPy Import --- #

# --- Prompts and Eval functions ---
PROMPT_TEMPLATE_CLASSES = {}
PROMPTS_AVAILABLE = False

# Placeholder for get_prompt_template to avoid early crash if src.prompts isn't found
def get_prompt_template_dummy(template_name, dataset_name, **kwargs):
    print(f"WARNING: Using dummy get_prompt_template for {template_name} on {dataset_name} due to earlier import failure.")
    class DummyPromptTemplate:
        def parse_output(self, **kwargs): return {"answer": None, "parsing_error": "Dummy parser due to src.prompts import fail"}
    return DummyPromptTemplate()

get_prompt_template = get_prompt_template_dummy


def _extract_number_robust(text: Optional[str]) -> Optional[float]:
    if text is None: return None
    text_str = str(text).strip().replace(",", "")
    if "####" in text_str: text_str = text_str.split("####")[-1].strip()
    match = re.search(r"[-+]?\d*\.?\d+", text_str)
    if match:
        try: return float(match.group(0))
        except ValueError: return None
    return None

def _is_correct_gsm8k_robust(result_item: Dict[str, Any], dataset_item: Dict[str, Any]) -> bool:
    true_answer_str = dataset_item.get("true_answer")
    predicted_answer_str = result_item.get("answer")
    if result_item.get("prompt_type") == "self_reflection":
        predicted_answer_str = result_item.get("final_answer_from_reflection", result_item.get("final_answer", predicted_answer_str))
    if predicted_answer_str is None or true_answer_str is None: return False
    true_num = _extract_number_robust(str(true_answer_str))
    predicted_num = _extract_number_robust(str(predicted_answer_str))
    if true_num is not None and predicted_num is not None: return math.isclose(true_num, predicted_num)
    return False

def _is_correct_strategyqa(result_item: Dict[str, Any], dataset_item: Dict[str, Any]) -> bool:
    true_answer_bool = dataset_item.get("true_answer")
    predicted_answer_val = result_item.get("answer")
    if result_item.get("prompt_type") == "self_reflection":
        predicted_answer_val = result_item.get("final_answer_from_reflection", result_item.get("final_answer", predicted_answer_val))
    if predicted_answer_val is None or true_answer_bool is None: return False
    if isinstance(predicted_answer_val, bool): return predicted_answer_val == true_answer_bool
    if isinstance(predicted_answer_val, str):
        pred_str_lower = predicted_answer_val.lower().strip()
        if pred_str_lower in ["yes", "true"]: return true_answer_bool is True
        if pred_str_lower in ["no", "false"]: return true_answer_bool is False
    return False

def _is_correct_math(result_item: Dict[str, Any], dataset_item: Dict[str, Any]) -> bool:
    is_math_eval_debug = False 
    # Example: uncomment and set ID to debug a specific item
    # if result_item.get("id") == "math_test_396.json": is_math_eval_debug = True
            
    if SYMPY_AVAILABLE:
        true_answer_str_orig = str(dataset_item.get("true_answer"))
        predicted_answer_val_orig = result_item.get("answer")
        
        # Handle self-reflection outputs for predicted answer
        if result_item.get("prompt_type") == "self_reflection":
            predicted_answer_val_orig = result_item.get("final_answer_from_reflection", 
                                                      result_item.get("final_answer", predicted_answer_val_orig))
        predicted_answer_str_orig = str(predicted_answer_val_orig)

        if is_math_eval_debug: 
            print(f"-- MATH DEBUG (Combined Plotter) ID: {result_item.get('id')} --")
            print(f"    Original True: '{true_answer_str_orig}', Original Pred: '{predicted_answer_str_orig}'")

        if not true_answer_str_orig or true_answer_str_orig.lower() == "none": 
            if is_math_eval_debug: print("    True answer string is empty/None. Eval False.")
            return False
        if not predicted_answer_str_orig or predicted_answer_str_orig.lower() == "none": 
            if is_math_eval_debug: print("    Predicted answer string is empty/None. Eval False.")
            return False

        true_answer_str = true_answer_str_orig
        predicted_answer_str = predicted_answer_str_orig

        true_boxed_match = re.search(r"\\boxed{(.+?)}", true_answer_str)
        if true_boxed_match: true_answer_str = true_boxed_match.group(1)
        pred_boxed_match = re.search(r"\\boxed{(.+?)}", predicted_answer_str)
        if pred_boxed_match: predicted_answer_str = pred_boxed_match.group(1)
        
        if is_math_eval_debug: print(f"    After boxed - True: '{true_answer_str}', Pred: '{predicted_answer_str}'")

        substitutions = { r"\\operatorname{(\w+)}": r"\\mathrm{\1}", r"\\xc2\\xb0": "", r"\\u00b0": "", r"%": "/100", r"\\cdot": "*", r"\\times": "*", r"\\div": "/"}
        for pat, repl in substitutions.items():
            true_answer_str = re.sub(pat, repl, true_answer_str)
            predicted_answer_str = re.sub(pat, repl, predicted_answer_str)
        true_answer_str = true_answer_str.replace("\\ ", " ").strip()
        predicted_answer_str = predicted_answer_str.replace("\\ ", " ").strip()

        if is_math_eval_debug: print(f"    After subs - True: '{true_answer_str}', Pred: '{predicted_answer_str}'")

        if not true_answer_str or not predicted_answer_str: 
            if is_math_eval_debug: print("    String became empty after processing. Eval False.")
            return False
        
        try:
            true_expr = parse_latex(true_answer_str)
            pred_expr = parse_latex(predicted_answer_str)
            if is_math_eval_debug: print(f"    Sympy Parsed - True: {true_expr}, Pred: {pred_expr}")

            if hasattr(true_expr, 'is_number') and true_expr.is_number and hasattr(pred_expr, 'is_number') and pred_expr.is_number:
                if math.isclose(N(true_expr), N(pred_expr), rel_tol=1e-3, abs_tol=1e-9): 
                    if is_math_eval_debug: print("    Numeric match FOUND.")
                    return True
            if hasattr(true_expr, 'equals') and true_expr.equals(pred_expr): 
                if is_math_eval_debug: print("    .equals() match FOUND.")
                return True
            
            # Ensure expressions are not None before subtraction
            if true_expr is not None and pred_expr is not None:
                if simplify(true_expr - pred_expr) == S.Zero: 
                    if is_math_eval_debug: print("    simplify(diff) == 0 match FOUND.")
                    return True
                if latex(simplify(true_expr)) == latex(simplify(pred_expr)): 
                    if is_math_eval_debug: print("    Simplified LaTeX match FOUND.")
                    return True
            if is_math_eval_debug: print("    All SymPy comparisons FAILED or not applicable.")
            return False
        except Exception as e:
            if is_math_eval_debug: print(f"    SymPy Processing Error: {type(e).__name__} - {e}. Falling back to string.")
            # Fall through to string comparison by not returning yet
            pass 

    # Fallback string comparison
    if is_math_eval_debug: print(f"    Attempting fallback string comparison for True: '{true_answer_str_orig}', Pred: '{predicted_answer_str_orig}'")
    norm_true = str(true_answer_str_orig).strip().lower().replace(" ", "") # Use original strings for fallback norm
    norm_pred = str(predicted_answer_str_orig).strip().lower().replace(" ", "")
    latex_cmds_to_remove = ["\\frac", "\\cdot", "\\left", "\\right", "(", ")", "{", "}", "\\text", "\\mathrm", "\\%", "%"]
    for cmd in latex_cmds_to_remove:
        norm_true = norm_true.replace(cmd, "")
        norm_pred = norm_pred.replace(cmd, "")
    if is_math_eval_debug: print(f"    Fallback Norm - True: '{norm_true}', Pred: '{norm_pred}'. Match: {norm_true == norm_pred}")
    return norm_true == norm_pred


def _is_correct_commonsenseqa(result_item: Dict[str, Any], dataset_item: Dict[str, Any]) -> bool:
    true_answer_label = dataset_item.get("true_answer")
    predicted_answer_val = result_item.get("answer")
    if result_item.get("prompt_type") == "self_reflection":
        predicted_answer_val = result_item.get("final_answer_from_reflection", result_item.get("final_answer", predicted_answer_val))
    if predicted_answer_val is None or true_answer_label is None: return False
    if isinstance(predicted_answer_val, str): return predicted_answer_val.strip().upper() == true_answer_label.strip().upper()
    return False

EVALUATION_FUNCTIONS = {
    "gsm8k": _is_correct_gsm8k_robust,
    "strategyqa": _is_correct_strategyqa,
    "math": _is_correct_math,
    "commonsenseqa": _is_correct_commonsenseqa,
}

def load_original_dataset_for_eval(dataset_name: str, split: str, data_dir_base: str) -> Dict[str, Dict[str, Any]]:
    data_path = Path(data_dir_base) / dataset_name
    raw_data_orig = []
    # ... (Copied and verified load_original_dataset_for_eval from regenerate_summary.py) ...
    if dataset_name == "gsm8k":
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("gsm8k", "main")
        for i, item in enumerate(ds[split if split == 'train' else 'test']): 
            raw_data_orig.append({"id": f"gsm8k_{split}_{i}", "question": item["question"], "true_answer": item["answer"].split("####")[-1].strip()})
    elif dataset_name == "math":
        math_split_dir = data_path / split
        if math_split_dir.is_dir():
            for subject_dir in math_split_dir.iterdir():
                if subject_dir.is_dir():
                    for problem_file in subject_dir.glob("*.json"):
                        try:
                            with open(problem_file, 'r', encoding='utf-8') as f: content = json.load(f)
                            solution = content.get("solution", "")
                            boxed_match = re.search(r"\\boxed{(.+?)}", solution)
                            answer = boxed_match.group(1).strip() if boxed_match else solution.strip()
                            raw_data_orig.append({"id": f"math_{split}_{problem_file.name}", "question": content.get("problem", ""), "true_answer": answer})
                        except Exception as e: print(f"Error loading MATH problem {problem_file}: {e}")
        else: print(f"MATH dataset directory not found: {math_split_dir}")
    elif dataset_name == "strategyqa":
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("ChilleD/StrategyQA") 
        for i, item in enumerate(ds[split]): 
            raw_data_orig.append({"id": f"strategyqa_{split}_{i}", "question": item["question"], "true_answer": item["answer"]})
    elif dataset_name == "commonsenseqa":
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("commonsense_qa")
        hf_split = "validation" if split == "test" or split == "dev" else split
        for i, item in enumerate(ds[hf_split]):
            question_text = item["question"]
            choices_text = "\\nChoices:\\n" + "\\n".join([f"({label}) {text}" for label, text in zip(item["choices"]["label"], item["choices"]["text"])])
            raw_data_orig.append({"id": f"csqa_{split}_{i}", "question": question_text + choices_text, "true_answer": item["answerKey"].upper()})
    
    return {item["id"]: item for item in raw_data_orig}


def load_process_and_categorize_items(
    model_name: str, 
    dataset_name: str, 
    data_dir_base: str = "data",
    results_dir_root: str = "results",
    force_reparse: bool = True 
    ):
    if not PROMPTS_AVAILABLE:
        print("ERROR: Prompt templates not available. Cannot proceed with parsing and evaluation.")
        return []

    dataset_split = "test" 
    original_dataset_items_map = load_original_dataset_for_eval(dataset_name, dataset_split, data_dir_base)
    if not original_dataset_items_map: 
        print(f"Critical Warning: Could not load original dataset for {dataset_name}.")

    results_dir_base_original = Path(results_dir_root) / dataset_name / model_name
    results_dir_base_alt = Path(results_dir_root) / dataset_name
    current_results_path_to_scan = results_dir_base_original
    filename_prefix_is_model_specific = False

    if dataset_name in ["strategyqa", "commonsenseqa"]:
        # For these datasets, prioritize checking the alternative path for model-prefixed files first
        # as per user's file structure screenshot.
        # Check if results_dir_base_alt contains files like <model_name>_<prompt_type>_...
        alt_path_has_model_files = False
        if results_dir_base_alt.exists():
            # Check for at least one file matching the model-prefix pattern
            if next(results_dir_base_alt.glob(f"{model_name}_*.json*"), None):
                alt_path_has_model_files = True

        if alt_path_has_model_files:
            print(f"INFO: For {dataset_name}/{model_name}, found model-prefixed files in alternative path: {results_dir_base_alt}. Using this path.")
            current_results_path_to_scan = results_dir_base_alt
            filename_prefix_is_model_specific = True
        elif results_dir_base_original.exists() and list(results_dir_base_original.glob("*.json*")):
            # If alt path didn't have model-prefixed files, but original model-specific path has files, use it.
            print(f"INFO: For {dataset_name}/{model_name}, using standard model-specific path: {results_dir_base_original}")
            current_results_path_to_scan = results_dir_base_original
            filename_prefix_is_model_specific = False # Files here are like <prompt_type>_...
        elif results_dir_base_alt.exists(): # Fallback if original is also empty, maybe non-prefixed files in alt?
            print(f"INFO: For {dataset_name}/{model_name}, standard path empty. Checking alt path {results_dir_base_alt} for non-prefixed files (less likely).")
            current_results_path_to_scan = results_dir_base_alt
            filename_prefix_is_model_specific = False # Assuming non-prefixed if we get here
        else:
            print(f"INFO: For {dataset_name}/{model_name}, neither standard nor alternative path yielded expected files. Using default: {current_results_path_to_scan}")
            # current_results_path_to_scan remains results_dir_base_original by default
    
    print(f"  Scanning for results in: {current_results_path_to_scan} (model prefix expected: {filename_prefix_is_model_specific})")
    processed_items_for_plot = []
    
    discovered_prompt_files: Dict[str, Path] = {}
    temp_prompt_files: Dict[str, List[Tuple[int, Path]]] = {}

    glob_patterns = ["*.json", "*.jsonl"]
    files_to_check_for_discovery = []
    for pattern in glob_patterns: files_to_check_for_discovery.extend(list(current_results_path_to_scan.glob(pattern)))
    files_to_check_for_discovery = sorted(list(set(files_to_check_for_discovery)))

    for f_path in files_to_check_for_discovery:
        for pt_class_name in PROMPT_TEMPLATE_CLASSES.keys():
            expected_prefix = pt_class_name + "_"
            model_specific_prefix = model_name + "_" + pt_class_name + "_"
            is_match = False
            if filename_prefix_is_model_specific and f_path.name.startswith(model_specific_prefix): is_match = True
            elif not filename_prefix_is_model_specific and f_path.name.startswith(expected_prefix): is_match = True
            if is_match:
                if pt_class_name not in temp_prompt_files: temp_prompt_files[pt_class_name] = []
                try: temp_prompt_files[pt_class_name].append((f_path.stat().st_size, f_path))
                except OSError: pass 
                break 
                
    for pt_name, files_with_sizes in temp_prompt_files.items():
        if files_with_sizes:
            largest_file_path = max(files_with_sizes, key=lambda x: x[0])[1]
            discovered_prompt_files[pt_name] = largest_file_path

    for prompt_type_name, file_path in discovered_prompt_files.items():
        print(f"    Loading from {prompt_type_name} file: {file_path.name}")
        raw_results_from_file = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.name.endswith(".jsonl"):
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try: raw_results_from_file.append(json.loads(line))
                            except json.JSONDecodeError as je: print(f"      JSONDecodeError in {file_path.name} line {line_num}: {je}")
                elif file_path.name.endswith(".json"):
                    content = json.load(f)
                    if isinstance(content, list): raw_results_from_file.extend(content)
                    elif isinstance(content, dict): raw_results_from_file.append(content)
        except Exception as e: print(f"      Error loading/parsing file {file_path.name}: {e}"); continue

        reparsed_items = []
        if force_reparse and PROMPTS_AVAILABLE:
            try:
                current_dataset_name_for_template = raw_results_from_file[0].get("dataset_name", dataset_name) if raw_results_from_file else dataset_name
                prompt_template_instance = get_prompt_template(prompt_type_name, dataset_name=current_dataset_name_for_template)
                for res_item_raw in raw_results_from_file:
                    parse_args = {"question": res_item_raw.get("question", "")}
                    raw_output_data = res_item_raw.get("raw_output")

                    if prompt_type_name == "self_consistency": 
                        parse_args["model_outputs"] = raw_output_data if isinstance(raw_output_data, list) else [str(raw_output_data)]
                        parsed_details = prompt_template_instance.parse_output(**parse_args)
                    elif prompt_type_name == "self_reflection":
                        # Correctly call SelfReflectionPrompt.parse_output with its defined parameter names
                        initial_out = res_item_raw.get("initial_reasoning", res_item_raw.get("raw_output"))
                        # Check if a distinct key for reflected output exists, otherwise it might be in a combined raw_output for some older formats
                        # Assuming 'reflection_reasoning' or a similar key holds the output after reflection if available.
                        # If not, this prompt type's re-parsing might be problematic if original JSON doesn't clearly separate stages.
                        reflected_out = res_item_raw.get("reflection_reasoning") # Or whatever key holds the second stage output
                        parsed_details = prompt_template_instance.parse_output(
                            model_output=initial_out, # First argument is model_output for initial pass
                            reflection_model_output=reflected_out, # Second is reflection_model_output
                            question=res_item_raw.get("question", "")
                        )
                    else: # For ZeroShot, FewShot, CoT, ReAct
                        parse_args["model_output"] = raw_output_data
                        parsed_details = prompt_template_instance.parse_output(**parse_args)
                    
                    updated_item = {**res_item_raw, **parsed_details, "prompt_type": prompt_type_name, "model": model_name, "dataset": dataset_name}
                    reparsed_items.append(updated_item)
            except Exception as e:
                print(f"      Error during re-parsing for {prompt_type_name} of {model_name}/{dataset_name}: {e}. Using items as-is.")
                reparsed_items = [{**item, "prompt_type": prompt_type_name, "model": model_name, "dataset": dataset_name} for item in raw_results_from_file]
        else: 
            reparsed_items = [{**item, "prompt_type": prompt_type_name, "model": model_name, "dataset": dataset_name} for item in raw_results_from_file]

        evaluation_function = EVALUATION_FUNCTIONS.get(dataset_name)
        for item in reparsed_items:
            original_item = original_dataset_items_map.get(item["id"])
            item_for_plot = item.copy() # Start with a copy
            if not original_item:
                item_for_plot["error_category"] = "Original_Item_Missing"
                item_for_plot["is_correct"] = False
            else:
                item_for_plot["true_answer"] = original_item.get("true_answer")
                is_correct = False
                if evaluation_function: is_correct = evaluation_function(item_for_plot, original_item) # Pass copy to eval
                item_for_plot["is_correct"] = is_correct
                if is_correct: item_for_plot["error_category"] = "Correct"
                else:
                    if item_for_plot.get("parsing_error") or item_for_plot.get("answer") is None: item_for_plot["error_category"] = "Parsing_Error_No_Answer"
                    else: item_for_plot["error_category"] = "Evaluation_Mismatch_UNKNOWN"
            processed_items_for_plot.append(item_for_plot)
    return processed_items_for_plot

def plot_combined_error_distribution(all_processed_data: List[Dict[str,Any]], figure_filename: str):
    if not all_processed_data:
        print("No data provided for combined error plot.")
        return

    df = pd.DataFrame(all_processed_data)
    if 'error_category' not in df.columns: df['error_category'] = 'UNKNOWN_CATEGORY'
    else: df['error_category'] = df['error_category'].fillna('UNKNOWN_CATEGORY')

    df['total_per_group'] = df.groupby(['model', 'dataset', 'prompt_type'])['id'].transform('count')
    df_counts = df.groupby(['model', 'dataset', 'prompt_type', 'error_category']).size().reset_index(name='count_error_category')
    df_totals_unique = df[['model', 'dataset', 'prompt_type', 'total_per_group']].drop_duplicates()
    df_plot_data = pd.merge(df_counts, df_totals_unique, on=['model', 'dataset', 'prompt_type'])
    df_plot_data['percentage'] = (df_plot_data['count_error_category'] / df_plot_data['total_per_group']) * 100

    dataset_order = ["gsm8k", "math", "strategyqa", "commonsenseqa"]
    model_order = sorted(list(df_plot_data['model'].unique())) # Ensure list for unique
    prompt_type_order = sorted(list(df_plot_data['prompt_type'].unique()))
    
    df_plot_data['dataset'] = pd.Categorical(df_plot_data['dataset'], categories=dataset_order, ordered=True)
    df_plot_data['prompt_type'] = pd.Categorical(df_plot_data['prompt_type'], categories=prompt_type_order, ordered=True)
    df_plot_data['model'] = pd.Categorical(df_plot_data['model'], categories=model_order, ordered=True)
    df_plot_data.sort_values(by=['model', 'dataset', 'prompt_type', 'error_category'], inplace=True) # Sort also by error_category for consistent stack order

    error_categories_sorted = sorted(list(df_plot_data['error_category'].unique()), key=lambda x: (x != 'Correct', x))
    
    # --- DEBUG PRINT --- Start
    print("\n--- DEBUG: df_plot_data for gpt-4/strategyqa ---")
    print(df_plot_data[(df_plot_data['model'] == 'gpt-4') & (df_plot_data['dataset'] == 'strategyqa')])
    print("--- END DEBUG ---\n")
    # --- DEBUG PRINT --- End

    # Create the FacetGrid
    g = sns.FacetGrid(df_plot_data, row='model', col='dataset', col_order=dataset_order, row_order=model_order, height=4, aspect=1.5, sharey=False)

    # Map a stacked bar plot to the grid
    def stacked_bar(data, color, **kwargs):
        # Pivot data for stacking within this facet
        pivot_df = data.pivot_table(index='prompt_type', columns='error_category', values='percentage', fill_value=0)
        # Reorder columns for consistent stack order
        cols_present = [col for col in error_categories_sorted if col in pivot_df.columns]
        pivot_df = pivot_df[cols_present]
        pivot_df.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='Spectral', legend=False, width=0.8)

    g.map_dataframe(stacked_bar)

    g.set_axis_labels("Prompt Type", "Percentage (%)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}") # Simpler titles
    g.fig.suptitle("Error Distribution: Model vs. Dataset vs. Prompt Type", y=1.03, fontsize=16)
    
    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelrotation=45) # Rotate x-axis labels
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Create a single legend for the entire figure
    # Get handles and labels from one of the subplots (if they were plotted with hue directly)
    # Since we are plotting manually in map_dataframe, create legend manually
    unique_error_categories = df_plot_data['error_category'].unique()
    # Get current palette
    palette = sns.color_palette('Spectral', n_colors=len(unique_error_categories))
    # Create proxy artists for legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=palette[i]) for i, _ in enumerate(unique_error_categories)]
    g.fig.legend(legend_handles, error_categories_sorted, title="Error Category", loc='center right', bbox_to_anchor=(1.17, 0.5))


    plt.tight_layout(rect=[0, 0, 0.88, 0.96]) 

    ensure_plot_dir()
    plot_path = Path("plots") / figure_filename
    plt.savefig(plot_path, dpi=300)
    print(f"Combined error plot saved to {plot_path}")
    plt.close(g.fig) 

def ensure_plot_dir():
    Path("plots").mkdir(parents=True, exist_ok=True)

def main():
    ensure_plot_dir() 
    
    # Make sure src.prompts can be imported
    global PROMPT_TEMPLATE_CLASSES, get_prompt_template, PROMPTS_AVAILABLE
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir # Assuming this script is in the project root. If not, adjust.
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(project_root)) 
    
    try:
        from src.prompts import PROMPT_TEMPLATE_CLASSES as PTC_imported, get_prompt_template as GPT_imported
        PROMPT_TEMPLATE_CLASSES = PTC_imported
        get_prompt_template = GPT_imported # Now correctly assigns the global
        PROMPTS_AVAILABLE = True
        print("INFO: src.prompts imported successfully in main().")
    except ImportError as e:
        print(f"ERROR: src.prompts could not be imported in main(): {e}. Using dummy versions. Plots may be incorrect or fail.")
        # Keep dummy versions defined at the top of the file
        PROMPTS_AVAILABLE = False # Ensure this is set if import fails

    models_to_plot = ["gpt-4", "deepseek-r1"]
    datasets_to_analyze = ["gsm8k", "math", "strategyqa", "commonsenseqa"]
    
    all_results_for_plotting = []

    for model in models_to_plot:
        for dataset in datasets_to_analyze:
            print(f"--- Processing data for: {model} on {dataset} ---")
            processed_data = load_process_and_categorize_items(
                model_name=model,
                dataset_name=dataset
            )
            all_results_for_plotting.extend(processed_data)
            print(f"Finished processing for: {model} on {dataset}. Processed {len(processed_data)} items.")

    if all_results_for_plotting:
        plot_combined_error_distribution(all_results_for_plotting, "combined_error_distribution_stacked.png")
    else:
        print("No data was processed. Combined error plot cannot be generated.")

if __name__ == "__main__":
    # Ensure the script can find the 'src' directory for imports.
    import sys
    # Assuming this script is in the project root alongside 'src/'
    # If it's in a subdirectory, adjust the path.
    # For example, if in 'scripts/', use Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main() 