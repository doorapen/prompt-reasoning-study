"""
Decoding strategies for LLMs, including self-consistency.
"""
from typing import List, Dict, Any
from collections import Counter

from src.prompts import PromptTemplate


def extract_answer(output: str) -> str:
    """Extract the final answer from the output."""
    # This is a simplistic implementation; you'll need to adapt it for each benchmark
    lines = output.strip().split('\n')
    return lines[-1].strip()


def majority_vote(outputs: List[str], prompt_template: PromptTemplate) -> str:
    """
    Implement majority voting for self-consistency.
    
    Args:
        outputs: List of model outputs from different samples
        prompt_template: The prompt template used for generation
        
    Returns:
        The majority voted answer
    """
    # Parse each output to extract the answer
    answers = []
    for output in outputs:
        parsed = prompt_template.parse_output(output)
        answers.append(parsed["answer"])
    
    # Count occurrences of each answer
    counter = Counter(answers)
    
    # Get the most common answer
    most_common_answer, count = counter.most_common(1)[0]
    
    return most_common_answer


def compute_self_consistency_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute metrics for self-consistency results.
    
    Args:
        results: List of results from self-consistency runs
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        "total": len(results),
        "correct": 0,
        "agreement_rates": [],
        "majority_correct": 0
    }
    
    for result in results:
        if "sample_outputs" not in result:
            continue
        
        # Count number of different answers
        answers = [extract_answer(output) for output in result["sample_outputs"]]
        counter = Counter(answers)
        
        # Calculate agreement rate (percentage of samples with the majority answer)
        most_common_answer, count = counter.most_common(1)[0]
        agreement_rate = count / len(answers)
        metrics["agreement_rates"].append(agreement_rate)
        
        # Check if majority answer is correct
        if most_common_answer == result["true_answer"]:
            metrics["majority_correct"] += 1
        
        # Check if any sample got the correct answer
        if result["true_answer"] in answers:
            metrics["correct"] += 1
    
    # Calculate averages
    if metrics["agreement_rates"]:
        metrics["avg_agreement_rate"] = sum(metrics["agreement_rates"]) / len(metrics["agreement_rates"])
    
    metrics["majority_accuracy"] = metrics["majority_correct"] / metrics["total"] if metrics["total"] > 0 else 0
    metrics["any_correct_accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
    
    return metrics

async def self_consistency(model, prompt_fn, task, k=5):
    outs = [await model(prompt_fn(task)) for _ in range(k)]
    answers = [parse.final_ans(o["text"]) for o in outs]
    maj = max(set(answers), key=answers.count)
    return {"text": "\n\n---\n".join(o["text"] for o in outs),
            "derived_answer": maj,
            "all_answers": answers}

async def self_reflection(model, cot_prompt, task):
    first = await model(cot_prompt(task))
    cot_text = first["text"]
    reflection_prompt = f"""{cot_text}

Is your answer above correct? Why or why not?
If incorrect, provide the correct answer in the format:
Final Answer: <answer>"""
    second = await model(reflection_prompt)
    return {"cot": first, "reflection": second} 