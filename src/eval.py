"""
Evaluation metrics for reasoning benchmarks.
"""
from typing import List, Dict, Any, Callable, Optional
import re
import numpy as np
from collections import Counter


def exact_match(prediction: str, reference: str) -> bool:
    """Check if prediction exactly matches reference."""
    return prediction.strip() == reference.strip()


def normalize_answer(text: str) -> str:
    """Normalize answer text for more robust matching."""
    # Remove punctuation and whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()


def contains_answer(prediction: str, reference: str) -> bool:
    """Check if prediction contains the reference answer."""
    return normalize_answer(reference) in normalize_answer(prediction)


def extract_number(text: str) -> Optional[float]:
    """Extract the first number from text."""
    match = re.search(r'[-+]?\d*\.\d+|\d+', text)
    if match:
        return float(match.group())
    return None


def numerical_match(prediction: str, reference: str, tolerance: float = 1e-6) -> bool:
    """Check if the numerical values match within tolerance."""
    pred_num = extract_number(prediction)
    ref_num = extract_number(reference)
    
    if pred_num is None or ref_num is None:
        return False
    
    # Check if within relative tolerance
    if abs(pred_num - ref_num) <= tolerance * max(1, abs(ref_num)):
        return True
    return False


def evaluate_gsm8k(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate results on GSM8K benchmark.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "total": len(results),
        "correct": 0,
        "error_types": Counter()
    }
    
    for result in results:
        prediction = result.get("answer", "")
        reference = result.get("true_answer", "")
        
        # GSM8K often contains numerical answers
        if numerical_match(prediction, reference):
            metrics["correct"] += 1
        else:
            # If there's an error code, add it to error types
            error_type = result.get("error_type", "UNKNOWN")
            metrics["error_types"][error_type] += 1
    
    metrics["accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
    
    return metrics


def evaluate_strategyqa(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate results on StrategyQA benchmark.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "total": len(results),
        "correct": 0,
        "error_types": Counter()
    }
    
    for result in results:
        prediction = result.get("answer", "").lower().strip()
        reference = result.get("true_answer", "").lower().strip()
        
        # StrategyQA has yes/no answers
        # Clean up the prediction to extract just the yes/no
        prediction_clean = re.sub(r'.*?(yes|no).*', r'\1', prediction)
        if prediction_clean not in ['yes', 'no']:
            # If we couldn't extract a clear yes/no, use the original
            prediction_clean = prediction
        
        if prediction_clean == reference:
            metrics["correct"] += 1
        else:
            # If there's an error code, add it to error types
            error_type = result.get("error_type", "UNKNOWN")
            metrics["error_types"][error_type] += 1
    
    metrics["accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
    
    return metrics


def bootstrap_ci(
    results: List[Dict[str, Any]],
    metric_fn: Callable[[List[Dict[str, Any]]], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Compute bootstrap confidence intervals for a metric.
    
    Args:
        results: List of result dictionaries
        metric_fn: Function that computes a metric from results
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Dictionary with metric value and confidence interval
    """
    # Compute the metric on the full dataset
    point_estimate = metric_fn(results)
    
    # Generate bootstrap samples and compute the metric on each
    bootstrap_estimates = []
    n = len(results)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, n, replace=True)
        bootstrap_sample = [results[i] for i in indices]
        
        # Compute metric on bootstrap sample
        bootstrap_metric = metric_fn(bootstrap_sample)
        bootstrap_estimates.append(bootstrap_metric)
    
    # Compute confidence interval
    alpha = (1 - confidence) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    
    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)
    
    return {
        "estimate": point_estimate,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "samples": bootstrap_estimates
    }


def mcnemar_test(results1: List[Dict[str, Any]], results2: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform McNemar's test to compare two methods.
    
    Args:
        results1: Results from method 1
        results2: Results from method 2
        
    Returns:
        Dictionary with test results
    """
    from scipy import stats
    
    # Group results by question ID
    results1_by_id = {r["id"]: r for r in results1}
    results2_by_id = {r["id"]: r for r in results2}
    
    # Only consider examples that both methods attempted
    common_ids = set(results1_by_id.keys()) & set(results2_by_id.keys())
    
    # Initialize contingency table
    # [method1 correct & method2 correct, method1 correct & method2 wrong]
    # [method1 wrong & method2 correct, method1 wrong & method2 wrong]
    contingency = [[0, 0], [0, 0]]
    
    for example_id in common_ids:
        res1 = results1_by_id[example_id]
        res2 = results2_by_id[example_id]
        
        correct1 = exact_match(res1.get("answer", ""), res1.get("true_answer", ""))
        correct2 = exact_match(res2.get("answer", ""), res2.get("true_answer", ""))
        
        if correct1 and correct2:
            contingency[0][0] += 1
        elif correct1 and not correct2:
            contingency[0][1] += 1
        elif not correct1 and correct2:
            contingency[1][0] += 1
        else:  # both wrong
            contingency[1][1] += 1
    
    # Calculate McNemar's test statistic
    # We're interested in the discordant pairs: method1 correct/method2 wrong vs method1 wrong/method2 correct
    b = contingency[0][1]  # method1 correct, method2 wrong
    c = contingency[1][0]  # method1 wrong, method2 correct
    
    # Use McNemar's test with continuity correction
    statistic = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
    p_value = stats.chi2.sf(statistic, 1)  # Chi-squared distribution with 1 degree of freedom
    
    return {
        "test_statistic": statistic,
        "p_value": p_value,
        "contingency_table": contingency,
        "method1_only_correct": b,
        "method2_only_correct": c,
        "both_correct": contingency[0][0],
        "both_wrong": contingency[1][1],
        "significant": p_value < 0.05
    } 