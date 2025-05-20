# generate_combined_length_correctness_plots.py
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

# --- SymPy Import ---
SYMPY_AVAILABLE = False
try:
    from sympy import sympify, simplify, N, latex, S
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
    print("INFO: generate_combined_length_correctness_plots.py: SymPy imported successfully.")
except ImportError:
    print("WARNING: generate_combined_length_correctness_plots.py: SymPy Import Error. MATH eval will use string fallback.")
# --- End SymPy Import ---

# --- Prompts and Eval functions ---
PROMPT_TEMPLATE_CLASSES = {}
PROMPTS_AVAILABLE = False
get_prompt_template = None

# Try to import from src.prompts AT THE TOP LEVEL for global availability
# This sys.path manipulation should ideally be done once if running script directly
import sys
script_dir_for_import = Path(__file__).resolve().parent
project_root_for_import = script_dir_for_import 
src_path_for_import = project_root_for_import / "src"
if str(project_root_for_import) not in sys.path: # Add project root to find src.
    sys.path.insert(0, str(project_root_for_import))

try:
    from src.prompts import PROMPT_TEMPLATE_CLASSES as PTC_imported, get_prompt_template as GPT_imported
    PROMPT_TEMPLATE_CLASSES = PTC_imported
    get_prompt_template = GPT_imported
    PROMPTS_AVAILABLE = True
    print("INFO: src.prompts imported successfully at top level for length_correctness_plot script.")
except ImportError as e:
    print(f"ERROR: src.prompts could not be imported at top level for length_correctness_plot script: {e}. Using dummy versions.")
    def get_prompt_template_dummy(template_name, dataset_name, **kwargs):
        # print(f"WARNING: Using dummy get_prompt_template for {template_name} on {dataset_name}.")
        class DummyPromptTemplate:
            def parse_output(self, **kwargs): return {"answer": None, "parsing_error": "Dummy parser due to src.prompts import fail"}
        return DummyPromptTemplate()
    get_prompt_template = get_prompt_template_dummy
    PROMPTS_AVAILABLE = False

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
    # if result_item.get("id") == "math_test_396.json": is_math_eval_debug = True
            
    if SYMPY_AVAILABLE:
        true_answer_str_orig = str(dataset_item.get("true_answer"))
        predicted_answer_val_orig = result_item.get("answer")
        if result_item.get("prompt_type") == "self_reflection":
            predicted_answer_val_orig = result_item.get("final_answer_from_reflection", result_item.get("final_answer", predicted_answer_val_orig))
        predicted_answer_str_orig = str(predicted_answer_val_orig)

        if not true_answer_str_orig or true_answer_str_orig.lower() == "none": return False
        if not predicted_answer_str_orig or predicted_answer_str_orig.lower() == "none": return False
        
        true_answer_str = true_answer_str_orig
        predicted_answer_str = predicted_answer_str_orig
        true_boxed_match = re.search(r"\\boxed{(.+?)}", true_answer_str)
        if true_boxed_match: true_answer_str = true_boxed_match.group(1)
        pred_boxed_match = re.search(r"\\boxed{(.+?)}", predicted_answer_str)
        if pred_boxed_match: predicted_answer_str = pred_boxed_match.group(1)
        
        substitutions = { r"\\operatorname{(\w+)}": r"\\mathrm{\1}", r"\\xc2\\xb0": "", r"\\u00b0": "", r"%": "/100", r"\\cdot": "*", r"\\times": "*", r"\\div": "/"}
        for pat, repl in substitutions.items():
            true_answer_str = re.sub(pat, repl, true_answer_str)
            predicted_answer_str = re.sub(pat, repl, predicted_answer_str)
        true_answer_str = true_answer_str.replace("\\ ", " ").strip()
        predicted_answer_str = predicted_answer_str.replace("\\ ", " ").strip()

        if not true_answer_str or not predicted_answer_str: return False
        
        try:
            true_expr = parse_latex(true_answer_str)
            pred_expr = parse_latex(predicted_answer_str)
            if hasattr(true_expr, 'is_number') and true_expr.is_number and hasattr(pred_expr, 'is_number') and pred_expr.is_number:
                if math.isclose(N(true_expr), N(pred_expr), rel_tol=1e-3, abs_tol=1e-9): return True
            if hasattr(true_expr, 'equals') and true_expr.equals(pred_expr): return True
            if true_expr is not None and pred_expr is not None:
                if simplify(true_expr - pred_expr) == S.Zero: return True
                if latex(simplify(true_expr)) == latex(simplify(pred_expr)): return True
            return False
        except Exception: pass 
    
    norm_true = str(true_answer_str_orig).strip().lower().replace(" ", "")
    norm_pred = str(predicted_answer_str_orig).strip().lower().replace(" ", "")
    latex_cmds_to_remove = ["\\frac", "\\cdot", "\\left", "\\right", "(", ")", "{", "}", "\\text", "\\mathrm", "\\%", "%"]
    for cmd in latex_cmds_to_remove:
        norm_true = norm_true.replace(cmd, "")
        norm_pred = norm_pred.replace(cmd, "")
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
    # ... (implementation as before, ensure datasets module is imported if used here, e.g. from datasets import load_dataset as hf_load_dataset ) ...
    data_path = Path(data_dir_base) / dataset_name
    raw_data_orig = []
    # Ensure `from datasets import load_dataset as hf_load_dataset` is at top or here if not already.
    try: from datasets import load_dataset as hf_load_dataset
    except ImportError: print("WARNING: `datasets` library not found, cannot load hf datasets in load_original_dataset_for_eval."); return {}

    if dataset_name == "gsm8k":
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
        ds = hf_load_dataset("ChilleD/StrategyQA") 
        for i, item in enumerate(ds[split]): 
            raw_data_orig.append({"id": f"strategyqa_{split}_{i}", "question": item["question"], "true_answer": item["answer"]})
    elif dataset_name == "commonsenseqa":
        ds = hf_load_dataset("commonsense_qa")
        hf_split = "validation" if split == "test" or split == "dev" else split
        for i, item in enumerate(ds[hf_split]):
            question_text = item["question"]
            choices_text = "\\nChoices:\\n" + "\\n".join([f"({label}) {text}" for label, text in zip(item["choices"]["label"], item["choices"]["text"])])
            raw_data_orig.append({"id": f"csqa_{split}_{i}", "question": question_text + choices_text, "true_answer": item["answerKey"].upper()})
    return {item["id"]: item for item in raw_data_orig}


def load_process_for_length_metrics(
    model_name: str, 
    dataset_name: str, 
    data_dir_base: str = "data",
    results_dir_root: str = "results",
    force_reparse: bool = True 
    ):
    if not PROMPTS_AVAILABLE:
        print(f"ERROR: Prompt templates not available for {model_name}/{dataset_name}. Cannot proceed.")
        return []

    dataset_split = "test" 
    original_dataset_items_map = load_original_dataset_for_eval(dataset_name, dataset_split, data_dir_base)
    if not original_dataset_items_map: 
        print(f"Critical Warning: Could not load original dataset for {dataset_name} for {model_name}.")

    results_dir_base_original = Path(results_dir_root) / dataset_name / model_name
    results_dir_base_alt = Path(results_dir_root) / dataset_name
    current_results_path_to_scan = results_dir_base_original
    filename_prefix_is_model_specific = False
    if dataset_name in ["strategyqa", "commonsenseqa"]:
        alt_path_has_model_files = False
        if results_dir_base_alt.exists():
            if next(results_dir_base_alt.glob(f"{model_name}_*.json*"), None):
                alt_path_has_model_files = True
        if alt_path_has_model_files:
            print(f"INFO: For {dataset_name}/{model_name}, found model-prefixed files in alternative path: {results_dir_base_alt}. Using this path.")
            current_results_path_to_scan = results_dir_base_alt
            filename_prefix_is_model_specific = True
        elif results_dir_base_original.exists() and list(results_dir_base_original.glob("*.json*")):
            print(f"INFO: For {dataset_name}/{model_name}, using standard model-specific path: {results_dir_base_original}")
            pass 
        elif results_dir_base_alt.exists():
            print(f"INFO: For {dataset_name}/{model_name}, standard path empty/missing. Checking alt path {results_dir_base_alt} for non-prefixed files.")
            current_results_path_to_scan = results_dir_base_alt
            filename_prefix_is_model_specific = False 
        else:
            print(f"INFO: For {dataset_name}/{model_name}, neither standard nor alternative path yielded expected files. Using default: {current_results_path_to_scan}")
    print(f"  Scanning for results in: {current_results_path_to_scan} (model prefix expected: {filename_prefix_is_model_specific})")
    items_for_plot = []
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
                        parse_args["model_outputs"] = raw_output_data if isinstance(raw_output_data, list) else ([str(raw_output_data)] if raw_output_data is not None else [])
                        parsed_details = prompt_template_instance.parse_output(**parse_args)
                    elif prompt_type_name == "self_reflection":
                        initial_out = res_item_raw.get("initial_reasoning", res_item_raw.get("raw_output"))
                        reflected_out = res_item_raw.get("reflection_reasoning") 
                        parsed_details = prompt_template_instance.parse_output(model_output=initial_out, reflection_model_output=reflected_out, question=res_item_raw.get("question", ""))
                    else: 
                        parse_args["model_output"] = raw_output_data
                        parsed_details = prompt_template_instance.parse_output(**parse_args)
                    updated_item = {**res_item_raw, **parsed_details, "prompt_type": prompt_type_name, "model": model_name, "dataset": dataset_name}
                    reparsed_items.append(updated_item)
            except Exception as e:
                print(f"      Error during re-parsing for {prompt_type_name} of {model_name}/{dataset_name}: {type(e).__name__} - {e}. Using items as-is.")
                reparsed_items = [{**item, "prompt_type": prompt_type_name, "model": model_name, "dataset": dataset_name} for item in raw_results_from_file]
        else: 
            reparsed_items = [{**item, "prompt_type": prompt_type_name, "model": model_name, "dataset": dataset_name} for item in raw_results_from_file]

        evaluation_function = EVALUATION_FUNCTIONS.get(dataset_name)
        for item in reparsed_items:
            original_item = original_dataset_items_map.get(item["id"])
            item_for_plot = item.copy()
            if not original_item:
                item_for_plot["is_correct"] = False
            else:
                item_for_plot["true_answer"] = original_item.get("true_answer")
                item_for_plot["is_correct"] = evaluation_function(item_for_plot, original_item) if evaluation_function else False
            
            # Calculate lengths
            explicit_rationale_text = item_for_plot.get("reasoning", "") # Prefer 'reasoning' for explicit rationale
            item_for_plot["explicit_rationale_length"] = len(str(explicit_rationale_text).split()) if explicit_rationale_text else 0
            
            model_full_output_text = item_for_plot.get("model_output_full", "")
            item_for_plot["total_output_length"] = len(str(model_full_output_text).split()) if model_full_output_text else 0
            
            items_for_plot.append(item_for_plot)
            
    return items_for_plot

def plot_combined_length_analysis(all_processed_data: List[Dict[str,Any]], length_column_name: str, y_label: str, figure_filename: str):
    if not all_processed_data:
        print(f"No data provided for {figure_filename}.")
        return

    df = pd.DataFrame(all_processed_data)
    if 'is_correct' not in df.columns or length_column_name not in df.columns:
        print(f"Missing required columns ('is_correct', '{length_column_name}') for {figure_filename}.")
        return
    
    # Filter out rows where the specific length column might be 0 if that means no relevant text was found
    # This is particularly important for 'explicit_rationale_length' which might be 0 for non-CoT prompts
    if length_column_name == "explicit_rationale_length":
        df = df[df["explicit_rationale_length"] > 0] # Only plot for prompts that have a rationale
        if df.empty:
            print(f"No data with explicit_rationale_length > 0 found for {figure_filename}.")
            return
            
    df['Correctness'] = df['is_correct'].apply(lambda x: "Correct" if x else "Incorrect")

    dataset_order = ["gsm8k", "math", "strategyqa", "commonsenseqa"]
    # Filter df for datasets present in the data to avoid empty categories in plot
    actual_datasets = [d for d in dataset_order if d in df['dataset'].unique()]
    if not actual_datasets: print("No matching datasets found in data for plotting."); return
    
    model_order = sorted(list(df['model'].unique()))
    prompt_type_order = sorted(list(df['prompt_type'].unique()))
    
    df['dataset'] = pd.Categorical(df['dataset'], categories=actual_datasets, ordered=True)
    df['prompt_type'] = pd.Categorical(df['prompt_type'], categories=prompt_type_order, ordered=True)
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    df.sort_values(by=['model', 'dataset', 'prompt_type'], inplace=True)

    g = sns.FacetGrid(df, row='model', col='dataset', col_order=actual_datasets, row_order=model_order, 
                      height=4, aspect=1.5, sharey=False, margin_titles=True)
    
    g.map_dataframe(sns.boxplot, x='prompt_type', y=length_column_name, hue='Correctness', 
                    palette={"Correct": "#66B2FF", "Incorrect": "#FF9999"}, dodge=True, medianprops=dict(color="black"))

    g.set_axis_labels("Prompt Type", y_label)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.fig.suptitle(f"{y_label} vs. Correctness by Model, Dataset, and Prompt Type", y=1.03, fontsize=16)
    
    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelrotation=45, labelsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    g.add_legend(title="Correctness", loc='upper right', bbox_to_anchor=(1.0, 0.95)) 
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    ensure_plot_dir()
    plot_path = Path("plots") / figure_filename
    plt.savefig(plot_path, dpi=300)
    print(f"Combined plot saved to {plot_path}")
    plt.close(g.fig)

def ensure_plot_dir():
    Path("plots").mkdir(parents=True, exist_ok=True)

def main():
    ensure_plot_dir() 
    global PROMPT_TEMPLATE_CLASSES, get_prompt_template, PROMPTS_AVAILABLE
    # ... (sys.path and src.prompts import logic as before) ...
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir 
    src_path = project_root / "src"
    if str(src_path) not in sys.path: sys.path.insert(0, str(project_root))
    try:
        from src.prompts import PROMPT_TEMPLATE_CLASSES as PTC_imported, get_prompt_template as GPT_imported
        PROMPT_TEMPLATE_CLASSES = PTC_imported
        get_prompt_template = GPT_imported
        PROMPTS_AVAILABLE = True
        print("INFO: src.prompts imported successfully in main() for length_metrics_plot script.")
    except ImportError as e:
        print(f"ERROR: src.prompts could not be imported in main() for length_metrics_plot script: {e}.")
        PROMPTS_AVAILABLE = False

    models_to_plot = ["gpt-4", "deepseek-r1"]
    datasets_to_analyze = ["gsm8k", "math", "strategyqa", "commonsenseqa"]
    all_results_for_plotting = []
    for model in models_to_plot:
        for dataset in datasets_to_analyze:
            print(f"--- Processing data for Length Metrics: {model} on {dataset} ---")
            # Renamed processing function
            processed_data = load_process_for_length_metrics(model_name=model, dataset_name=dataset)
            all_results_for_plotting.extend(processed_data)
            print(f"Finished Length Metrics processing for: {model} on {dataset}. Processed {len(processed_data)} items.")

    if all_results_for_plotting:
        # Plot for Explicit Rationale Length
        plot_combined_length_analysis(all_results_for_plotting, 
                                      length_column_name="explicit_rationale_length", 
                                      y_label="Explicit Rationale Length (Words)", 
                                      figure_filename="combined_explicit_rationale_length_vs_correctness.png")
        # Plot for Total Output Length
        plot_combined_length_analysis(all_results_for_plotting, 
                                      length_column_name="total_output_length", 
                                      y_label="Total Output Length (Words)", 
                                      figure_filename="combined_total_output_length_vs_correctness.png")
    else:
        print("No data was processed. Length vs. Correctness plots cannot be generated.")

if __name__ == "__main__":
    main() 