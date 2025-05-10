"""
Prompt templates for different reasoning strategies.
"""
from typing import List, Dict, Any, Optional
import os
import re
from collections import Counter

class PromptTemplate:
    """Base class for all prompt templates."""
    
    def __init__(self, name: str):
        self.name = name
    
    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        """Format the prompt template with the task input."""
        raise NotImplementedError
    
    def _extract_answer(self, text: str) -> Optional[float]:
        """
        Extracts the numerical answer from a single model output string.
        Tries "#### <number>", then specific phrases, then the last number found.
        """
        if text is None:
            return None
        
        text_str = str(text).strip()
        # Standardize by removing commas that might be in numbers
        text_str_no_commas = text_str.replace(",", "")

        # Priority 1: "#### <number>"
        # Take the part after the LAST "####"
        if "####" in text_str_no_commas:
            parts = text_str_no_commas.split("####")
            answer_part = parts[-1].strip()
            # Find the first number in this part
            match = re.search(r"[-+]?\d*\.?\d+", answer_part)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    pass # Fall through if conversion fails, try other methods

        # Priority 2: Look for specific phrases like "The final answer is X"
        # This regex looks for variations of "The final answer is", "The answer is", etc., followed by a number.
        # It's applied to the text_str_no_commas for consistency.
        specific_phrase_match = re.search(
            r"(?:The final answer is|The answer is|Therefore, the answer is|So the answer is)\s*[:\s]*([-+]?\d*\.?\d+)", 
            text_str_no_commas, 
            re.IGNORECASE
        )
        if specific_phrase_match:
            try:
                return float(specific_phrase_match.group(1))
            except ValueError:
                pass # Fall through

        # Priority 3 (General fallback): Find all numbers in text_str_no_commas and return the last one.
        # This is a common heuristic for GSM8K if no other specific pattern matches.
        numbers_found = re.findall(r"[-+]?\d*\.?\d+", text_str_no_commas)
        if numbers_found:
            try:
                # Get the last number found in the string
                return float(numbers_found[-1])
            except (ValueError, IndexError):
                # Should not happen if numbers_found is populated and they are valid numbers
                return None 
        
        return None # Default if no answer found

    def parse_output(self, model_output: str, *args) -> Dict[str, Any]:
        """
        Parses the model output string and extracts the answer.
        (This is the method for single outputs, used by ZeroShot, FewShot etc.)
        """
        answer = self._extract_answer(model_output)
        if answer is not None:
            return {"answer": answer}
        else:
            return {"answer": None, "parsing_error": "Could not extract answer from model output"}


class ZeroShotPrompt(PromptTemplate):
    """Zero-shot prompt template."""
    
    def __init__(self):
        super().__init__("zero_shot")
    
    def format(self, question: str, examples: List[Dict[str, str]] = None) -> str:
        # More structured prompt that works for all models
        return f"""Solve the following problem:
{question}

Make sure to show your work step-by-step, and finish with the final answer clearly stated as:
"Therefore, the answer is: [your answer]"
"""
    
    def parse_output(self, output: str) -> Dict[str, Any]:
        reasoning_text = output # Default reasoning is the full output
        answer_val = None
        parsing_error = None

        # Try specific pattern first, as instructed by the prompt:
        # "Therefore, the answer is: [your answer]"
        # This regex looks for variations of that phrase.
        specific_pattern = r"(?:Therefore,\s*the\s*answer\s*is|The\s*final\s*answer\s*is)\s*[:\s]*([-+]?\d*\.?\d+)"
        match = re.search(specific_pattern, output, re.IGNORECASE)
        
        if match:
            try:
                answer_val = float(match.group(1).replace(",", ""))
            except ValueError:
                # Found the phrase but the number part was invalid
                parsing_error = "Found specific answer phrase but number parsing failed."
        
        if answer_val is None: 
            # If specific pattern failed or wasn't found, fallback to the robust base parser
            parsed_base = super().parse_output(output) # Uses the enhanced _extract_answer
            answer_val = parsed_base.get("answer")
            if answer_val is None and not parsing_error: # If base also failed, and we didn't have a specific error yet
                parsing_error = parsed_base.get("parsing_error", "Could not extract answer using base parser.")
            # If base succeeded, clear any prior specific parsing error
            elif answer_val is not None:
                parsing_error = None
        
        final_parsing_error = parsing_error if answer_val is None else None

        return {"answer": answer_val, 
                "reasoning": reasoning_text, 
                "model_output_full": output,
                "parsing_error": final_parsing_error}


class FewShotPrompt(PromptTemplate):
    """Few-shot prompt template."""
    
    def __init__(self, num_examples: int = 3):
        super().__init__("few_shot")
        self.num_examples = num_examples
    
    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        if not examples or len(examples) < self.num_examples:
            raise ValueError(f"Need at least {self.num_examples} examples for few-shot prompting")
        
        prompt = ""
        for i, example in enumerate(examples[:self.num_examples]):
            prompt += f"Q: {example['question']}\nA: {example['answer']}\n\n"
        
        prompt += f"Q: {task_input}\nA:"
        return prompt
    
    # No parse_output override, so it will use the robust base PromptTemplate.parse_output


class ChainOfThoughtPrompt(PromptTemplate):
    """Chain-of-Thought prompt template."""
    
    def __init__(self):
        super().__init__("cot")
    
    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        return f"{task_input}\nLet's think step by step."
    
    def parse_output(self, output: str) -> Dict[str, Any]:
        # Use the robust base parser to get the numerical answer
        parsed_base = super().parse_output(output)
        answer_val = parsed_base.get("answer")

        reasoning_text = output 
        
        # If base parser found an error, reflect it
        parsing_error_to_return = parsed_base.get("parsing_error") if answer_val is None else None
            
        return {"answer": answer_val, 
                "reasoning": reasoning_text, 
                "model_output_full": output,
                "parsing_error": parsing_error_to_return}


class SelfConsistencyPrompt(ChainOfThoughtPrompt):
    """Self-Consistency prompt template (extends CoT)."""
    
    def __init__(self, num_samples: int = 5):
        super().__init__()
        self.name = "self_consistency"
        self.num_samples = num_samples

    def parse_output(self, sample_outputs: List[str], *args) -> Dict[str, Any]:
        """
        Parses a list of sample outputs, extracts an answer from each,
        and returns the majority vote.
        """
        if not sample_outputs or not isinstance(sample_outputs, list):
            return {"answer": None, "parsing_error": "No sample_outputs provided or not a list"}

        extracted_answers = []
        for single_output_str in sample_outputs:
            if not isinstance(single_output_str, str):
                # Optionally log a warning for non-string items
                # print(f"Warning: Non-string item in sample_outputs: {type(single_output_str)}")
                continue
            
            # Use the _extract_answer method (inherited or defined in PromptTemplate)
            # to parse each individual sample string.
            answer_val = self._extract_answer(single_output_str)
            
            if answer_val is not None:
                # Convert to string for reliable counting, especially if answers can be numbers/strings
                extracted_answers.append(str(answer_val))

        if not extracted_answers:
            return {"answer": None, "parsing_error": "No answers could be extracted from any sample_outputs"}

        # Perform majority vote
        answer_counts = Counter(extracted_answers)
        # most_common(1) returns a list of tuples [(element, count)], e.g., [('123', 3)]
        most_common_list = answer_counts.most_common(1)
        
        if not most_common_list:
            # This case should ideally not be reached if extracted_answers is populated
            return {"answer": None, "parsing_error": "Majority vote failed (no common answers)"}

        final_answer_str = most_common_list[0][0]
        
        # The final_answer_str is a string. If your _is_correct_gsm8k_robust
        # expects a number, it will handle the conversion via _extract_number_robust.
        # So, returning the string representation from the majority vote is fine.
        return {"answer": final_answer_str}


class SelfReflectionPrompt(PromptTemplate):
    """Self-Reflection prompt template."""
    
    def __init__(self):
        super().__init__("self_reflection")
    
    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        # For self-reflection, the initial prompt is often a CoT-style prompt
        return f"{task_input}\nLet's think step by step to get an initial answer."
    
    def format_reflection(self, initial_output: str) -> str:
        """Format the reflection prompt based on initial output."""
        # The reflection prompt should guide the model to critique and potentially correct its initial output.
        return f"Here is an initial attempt to solve the problem:\n{initial_output}\n\nPlease review this solution. Are there any errors? If so, correct them and provide the final answer. If the solution is correct, confirm it.\nFinal Answer:"

    def parse_output(self, initial_model_output: str, reflection_model_output: str) -> Dict[str, Any]:
        # Parse the initial output (e.g., using CoT logic or base logic)
        # For simplicity, let's assume initial_output might contain reasoning and an answer
        parsed_initial = super().parse_output(initial_model_output) # Use robust base for initial answer
        
        # Parse the reflection output for the final answer
        parsed_reflection_final = super().parse_output(reflection_model_output) # Use robust base for reflected answer
        
        final_answer_val = parsed_reflection_final.get("answer")
        
        # If reflection didn't yield a clear answer, consider the initial one (if valid)
        if final_answer_val is None and parsed_initial.get("answer") is not None:
            final_answer_val = parsed_initial.get("answer")

        return {
            "initial_output": initial_model_output,
            "reflection_output": reflection_model_output,
            "initial_answer_parsed": parsed_initial.get("answer"),
            "final_answer_from_reflection": parsed_reflection_final.get("answer"),
            "answer": final_answer_val, # This is the one used for evaluation
            "model_output_full": f"INITIAL:\n{initial_model_output}\n\nREFLECTION:\n{reflection_model_output}",
            "parsing_error_initial": parsed_initial.get("parsing_error"),
            "parsing_error_reflection": parsed_reflection_final.get("parsing_error")
        }


class ReActPrompt(PromptTemplate):
    """ReAct prompt template (Reasoning + Acting)."""
    
    def __init__(self):
        super().__init__("react")
    
    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        prompt = f"{task_input}\n\n"
        prompt += (
            "Let's solve this problem step by step. You can use the following actions:\n"
            "- Thought: think about the current state and what to do next\n"
            "- Action: Compute[expression] - compute the result of a mathematical expression\n"
            "- Answer: [your final answer]\n\n"
            "Start with a Thought."
        )
        return prompt
    
    def parse_output(self, output: str) -> Dict[str, Any]:
        lines = output.strip().split('\n')
        thoughts = []
        actions = []
        parsed_answer_val = None 
        parsing_error_final = None
        
        # Try to find "Answer:" line first
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("Thought:"):
                thoughts.append(line_stripped[len("Thought:"):].strip())
            elif line_stripped.startswith("Action: Compute["):
                expr = line_stripped[len("Action: Compute["):]
                expr = expr.split("]")[0] if "]" in expr else expr
                actions.append(expr)
            elif line_stripped.startswith("Answer:"):
                answer_text_after_marker = line_stripped[len("Answer:"):].strip()
                # Try to parse this specific answer text robustly using the base method
                temp_parsed = super().parse_output(answer_text_after_marker) # Uses enhanced _extract_answer
                if temp_parsed.get("answer") is not None:
                    parsed_answer_val = temp_parsed.get("answer")
                    # If we found an answer here, clear any potential error from later fallbacks
                    parsing_error_final = None 
                    break 
                else:
                    # Store error from this attempt if it's the first one
                    if parsing_error_final is None:
                         parsing_error_final = temp_parsed.get("parsing_error", "Failed to parse content after 'Answer:' marker.")
        
        # If "Answer:" marker didn't yield an answer, try parsing the whole output
        if parsed_answer_val is None:
            parsed_base = super().parse_output(output) # Uses enhanced _extract_answer
            parsed_answer_val = parsed_base.get("answer")
            if parsed_answer_val is None:
                # If we already had an error from "Answer:" marker parsing, keep it, otherwise use base error
                parsing_error_final = parsing_error_final if parsing_error_final else parsed_base.get("parsing_error", "Failed to parse whole output.")
            else:
                # Base parsing succeeded, so no error.
                parsing_error_final = None


        result = {
            "model_output_full": output, 
            "thoughts": thoughts,
            "actions": actions,
            "answer": parsed_answer_val,
            "parsing_error": parsing_error_final if parsed_answer_val is None else None
        }
        return result


# Dictionary mapping prompt types to their classes
PROMPT_TEMPLATES = {
    "zero_shot": ZeroShotPrompt(),
    "few_shot": FewShotPrompt(),
    "cot": ChainOfThoughtPrompt(),
    "self_consistency": SelfConsistencyPrompt(),
    "self_reflection": SelfReflectionPrompt(),
    "react": ReActPrompt()
}


def get_prompt_template(prompt_type: str) -> PromptTemplate:
    """Get a prompt template by type."""
    if prompt_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt type: {prompt_type}. "
                         f"Available types: {list(PROMPT_TEMPLATES.keys())}")
    return PROMPT_TEMPLATES[prompt_type] 