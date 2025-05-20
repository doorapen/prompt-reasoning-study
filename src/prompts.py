"""
Prompt templates for different reasoning strategies.
"""
from typing import List, Dict, Any, Optional, Union
import os
import re
from collections import Counter
import random # Add import for random selection

# --- Few-Shot Examples ---
FEW_SHOT_EXAMPLES_GSM8K = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer_explanation": "Natalia sold 48 clips in April.\nIn May, she sold half as many clips as in April.\nSo, in May, she sold 48 / 2 = 24 clips.\nAltogether, Natalia sold 48 + 24 = 72 clips in April and May.\nThe final answer is #### 72"
    },
    # Add more GSM8K examples if you have them
]

FEW_SHOT_EXAMPLES_STRATEGYQA = [
    {
        "question": "Could a high school student be president of the United States?",
        "answer_explanation": "To be president of the United States, a person must meet certain constitutional requirements. One of these requirements is age: a person must be at least 35 years old. High school students are typically between 14 and 18 years old. Since 14-18 is less than 35, a high school student would not meet the age requirement. Therefore, a high school student could not be president of the United States.\nThe final answer is No."
    },
    {
        "question": "Is it possible for a cat to live longer than a human?",
        "answer_explanation": "The average lifespan of a domestic cat is around 12-15 years, with some living into their early 20s. The average human lifespan is significantly longer, typically 70-80 years or more depending on the region and other factors. While there might be exceptionally long-lived cats and short-lived humans, it is generally not possible for a cat to live longer than a human's typical lifespan. The record for the oldest cat is around 38 years, which is still much shorter than many humans live.\nThe final answer is No."
    },
    {
        "question": "Is it possible for the same person to be both a US Senator and a US Representative simultaneously?",
        "answer_explanation": "The US Constitution, specifically Article I, Section 6, Clause 2 (the Incompatibility Clause), states that 'no Person holding any Office under the United States, shall be a Member of either House during his Continuance in Office.' While a Senator and Representative hold office under the legislative branch, not the executive, the widely accepted interpretation and historical practice is that membership in the House and Senate are incompatible offices. One must resign from one position to serve in the other. Therefore, it is not possible for the same person to hold both positions at the same time.\nWait, re-reading the clause, it refers to holding an 'Office under the United States', which typically refers to executive or judicial posts, not membership in Congress itself. However, the structure of Congress and centuries of practice make it practically impossible. While perhaps not explicitly forbidden by that specific clause for Congressional seats themselves, the functional reality and rules of each house would prevent simultaneous service. Let's reconsider. Can a person *be elected* to both? Yes. Can they *serve* in both simultaneously? No. The question asks if it's *possible*. Given they can't serve simultaneously, the answer should be No.\nLet's try a different approach. If a person were elected to both, they would have to choose which seat to accept. They cannot be sworn into both offices. Thus, they cannot *be* both simultaneously in the functional sense of holding office.\nThe final answer is No."
    },
    {
        "question": "Can a vegetarian restaurant serve a cheeseburger?",
        "answer_explanation": "A vegetarian restaurant, by definition, does not serve meat, poultry, or fish. A cheeseburger traditionally contains a beef patty. While vegetarian burgers (made from plant-based ingredients) exist, a standard cheeseburger includes meat. Serving a meat-based cheeseburger would violate the core principle of a vegetarian establishment. Therefore, a vegetarian restaurant cannot serve a (traditional) cheeseburger.\nThe final answer is No."
    },
    {
        "question": "Can you use a standard pencil to write underwater?",
        "answer_explanation": "Standard pencils use graphite (mixed with a clay binder) as the writing core. Unlike ink pens which rely on ink flow that can be disrupted by water, graphite works by physically abrading onto the paper surface. This physical process does not require the absence of water. You can indeed write with a standard graphite pencil on suitable surfaces (like waterproof paper or plastics) underwater. The graphite will transfer successfully.\nThe final answer is Yes."
    },
    # Add 1-3 more high-quality StrategyQA examples here.
    # Ensure they show reasoning leading to a "Yes" or "No" answer.
]

FEW_SHOT_EXAMPLES_MATH = [
    {
        "question": "Find the derivative of $f(x) = x^3 - 6x^2 + 2x - 1$.",
        "answer_explanation": "To find the derivative of $f(x) = x^3 - 6x^2 + 2x - 1$, we use the power rule for differentiation, which states that the derivative of $x^n$ is $nx^{n-1}$.\nApplying this rule to each term:\nThe derivative of $x^3$ is $3x^{3-1} = 3x^2$.\nThe derivative of $-6x^2$ is $-6 \cdot 2x^{2-1} = -12x$.\nThe derivative of $2x$ is $2 \cdot 1x^{1-1} = 2x^0 = 2 \cdot 1 = 2$.\nThe derivative of a constant (like -1) is 0.\nSo, $f'(x) = 3x^2 - 12x + 2$.\nThe final answer is $\\boxed{3x^2 - 12x + 2}$"
    },
    {
        "question": "Solve the equation $2(x - 3) + 5 = 9$.",
        "answer_explanation": "To solve the equation $2(x - 3) + 5 = 9$, we first distribute the 2 on the left side:\n$2x - 2 \cdot 3 + 5 = 9$\n$2x - 6 + 5 = 9$\nCombine the constant terms on the left side:\n$2x - 1 = 9$\nAdd 1 to both sides of the equation:\n$2x - 1 + 1 = 9 + 1$\n$2x = 10$\nDivide both sides by 2:\n$\\frac{2x}{2} = \\frac{10}{2}$\n$x = 5$\nThe final answer is $\\boxed{5}$"
    },
    # Add 1-2 more diverse MATH examples if possible.
]

FEW_SHOT_EXAMPLES_COMMONSENSEQA = [
    {
        "question": "Where would a student typically go to borrow a book?\nChoices:\n(A) grocery store\n(B) library\n(C) movie theater\n(D) amusement park\n(E) gas station",
        "answer_explanation": "The question asks where a student would typically go to borrow a book. Let's consider the options:\n(A) A grocery store is for buying food.\n(B) A library is a place specifically designed for lending books.\n(C) A movie theater is for watching films.\n(D) An amusement park is for entertainment rides.\n(E) A gas station is for fueling vehicles.\nBased on these, a library is the most appropriate place to borrow a book.\nThe final answer is (B)."
    },
    {
        "question": "What is a common reason for a person to open a window in a stuffy room?\nChoices:\n(A) To make the room darker\n(B) To get fresh air\n(C) To make the room warmer\n(D) To practice their throwing skills\n(E) To hear the birds sing louder",
        "answer_explanation": "The question asks for a common reason to open a window in a stuffy room. A stuffy room usually implies poor air circulation or stale air. Let's evaluate the choices:\n(A) Opening a window usually makes a room brighter, not darker.\n(B) Getting fresh air is a primary reason to open a window, especially if a room is stuffy.\n(C) Opening a window often makes a room cooler, not warmer, unless it's warmer outside.\n(D) While possible, this is not a common or primary reason.\n(E) This might be a side effect, but not the main reason for opening a window in a stuffy room.\nTherefore, getting fresh air is the most common reason.\nThe final answer is (B)."
    },
    # Add 1-2 more CommonsenseQA examples. Ensure the question includes the choices.
]

ALL_FEW_SHOT_EXAMPLES = {
    "gsm8k": FEW_SHOT_EXAMPLES_GSM8K,
    "strategyqa": FEW_SHOT_EXAMPLES_STRATEGYQA,
    "math": FEW_SHOT_EXAMPLES_MATH,
    "commonsenseqa": FEW_SHOT_EXAMPLES_COMMONSENSEQA,
}

class PromptTemplate:
    """Base class for all prompt templates."""
    
    def __init__(self, name: str, system_prompt: Optional[str] = None, dataset_name: Optional[str] = None):
        self.name = name
        self.system_prompt = system_prompt
        self.dataset_name = dataset_name

        # Automatically load few_shot_examples if this is a FewShotPrompt and examples are available
        # This specific logic might be better placed in FewShotPrompt's __init__ if it's the only one using it.
        # For now, keeping it here for broader potential use, but it means dataset_name must be known.
        if self.name == "few_shot" and self.dataset_name and self.dataset_name in ALL_FEW_SHOT_EXAMPLES:
            self.few_shot_examples_for_dataset = ALL_FEW_SHOT_EXAMPLES[self.dataset_name]
        else:
            self.few_shot_examples_for_dataset = []
            
    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        """Format the prompt template with the task input."""
        raise NotImplementedError
    
    def _extract_answer_gsm8k(self, text: str) -> Optional[float]:
        """
        Extracts the numerical answer from a single model output string for GSM8K.
        Tries "#### <number>", then specific phrases, then the last number found.
        (This is your existing robust logic for GSM8K)
        """
        if text is None:
            return None
        
        text_str = str(text).strip()
        text_str_no_commas = text_str.replace(",", "")

        if "####" in text_str_no_commas:
            parts = text_str_no_commas.split("####")
            answer_part = parts[-1].strip()
            match = re.search(r"[-+]?\d*\.?\d+", answer_part)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    pass

        specific_phrase_match = re.search(
            r"(?:The final answer is|The answer is|Therefore, the answer is|So the answer is)\s*[:\s]*([-+]?\d*\.?\d+)", 
            text_str_no_commas, 
            re.IGNORECASE
        )
        if specific_phrase_match:
            try:
                return float(specific_phrase_match.group(1))
            except ValueError:
                pass

        numbers_found = re.findall(r"[-+]?\d*\.?\d+", text_str_no_commas)
        if numbers_found:
            try:
                return float(numbers_found[-1])
            except (ValueError, IndexError):
                return None 
        return None

    def _extract_answer_strategyqa(self, text: str) -> Optional[bool]:
        """
        Extracts a boolean answer (True/False) from StrategyQA model output.
        Looks for "Yes", "No", "True", "False" as the concluding part of the answer.
        Returns True for "Yes", False for "No".
        """
        if text is None:
            return None
        
        text_lower = text.lower().strip()

        # Try to find explicit "The final answer is Yes/No" or similar, near the end.
        # Regex to find "yes" or "no" possibly preceded by "the final answer is" or similar,
        # and not followed by much other text.
        # This looks for "yes" or "no" that are likely the concluding statement.
        
        # More robust: check the last sentence or last few words.
        # Split by common sentence delimiters or just take the last ~20 characters.
        # This is a heuristic.
        
        # Simplified: look for "yes" or "no" in the last part of the string.
        # Consider the last sentence or a small window at the end.
        # Example: "The final answer is Yes." or "So the answer is no."
        
        # Search for "yes" or "no" potentially qualified by "final answer is"
        match_yes = re.search(r"(?:the final answer is|the answer is|is simply|is:)\s*(yes|true)\b", text_lower)
        match_no = re.search(r"(?:the final answer is|the answer is|is simply|is:)\s*(no|false)\b", text_lower)

        if match_yes and match_no:
            # Ambiguous if both are found with "final answer is". Prefer later one.
            if match_yes.start() > match_no.start():
                return True
            else:
                return False
        elif match_yes:
            return True
        elif match_no:
            return False

        # Fallback: Check the very end of the string for "yes." or "no." or " yes" or " no"
        # This is more lenient.
        if text_lower.endswith("yes.") or text_lower.endswith("yes") or text_lower.endswith("true.") or text_lower.endswith("true"):
            return True
        if text_lower.endswith("no.") or text_lower.endswith("no") or text_lower.endswith("false.") or text_lower.endswith("false"):
            return False
        
        # Broader search if specific markers fail (can be risky)
        # Check last ~30 chars
        last_part = text_lower[-30:]
        if "yes" in last_part and "no" not in last_part: # Avoid "is not yes"
             if "not yes" not in last_part and "not true" not in last_part:
                return True
        if "no" in last_part and "yes" not in last_part:
             if "not no" not in last_part and "not false" not in last_part:
                return False
        
        return None # Cannot determine

    def _extract_answer_math(self, text: str) -> Optional[str]:
        """
        Extracts the LaTeX answer from a model's MATH output.
        Primarily looks for \\boxed{...}.
        If not found, might try to extract the last equation or a significant mathematical expression.
        """
        if text is None:
            return None
        text_str = str(text).strip()

        # Priority 1: Look for \boxed{...}
        # Using a non-greedy match for the content inside \boxed{}
        boxed_match = re.search(r"\\boxed{(.+?)}", text_str)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Priority 2: Look for "The final answer is <LaTeX/expression>"
        # This is more heuristic for MATH as answers are complex.
        final_answer_match = re.search(
            r"(?:The final answer is|The answer is)\s*[:\s]*(.+)", 
            text_str, 
            re.IGNORECASE
        )
        if final_answer_match:
            potential_answer = final_answer_match.group(1).strip()
            # Check if it looks like a math expression (heuristic)
            if '$' in potential_answer or '\\' in potential_answer or '=' in potential_answer or any(c.isdigit() for c in potential_answer):
                return potential_answer
        
        # Fallback: Return the last non-empty line if it seems like a formula.
        # This is a very basic fallback and might not be very accurate.
        lines = [line.strip() for line in text_str.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            # Heuristic: if it contains common math symbols or structure
            if '$' in last_line or '\\' in last_line or '=' in last_line or any(c.isdigit() for c in last_line):
                # Avoid returning very long lines that are likely full explanations
                if len(last_line) < 150: # Arbitrary length limit
                    return last_line
        return None

    def _extract_answer_commonsenseqa(self, text: str) -> Optional[str]:
        """
        Extracts the choice (e.g., "A", "B") from CommonsenseQA model output.
        Looks for patterns like "(A)", "Answer: A", "The answer is B".
        Returns the uppercase letter of the choice.
        """
        if text is None:
            return None
        text_str = str(text).strip()

        # Priority 1: Look for "The final answer is (X)" or "Answer: (X)" or "Answer: X"
        # where X is a single letter A-E.
        # Regex captures the letter.
        # It allows for optional parentheses around the letter.
        # It looks for common preceding phrases.
        match = re.search(
            r"(?:The final answer is|The answer is|Answer is|My answer is|Final Answer:)\s*\(?([A-Ea-e])\)?", 
            text_str, 
            re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # Priority 2: Look for just "(X)" or "X)" or "X." at the end of the string or a line.
        # This is more general.
        # Try to find (A), (B)... at the end of the text or last significant line.
        lines = [line.strip() for line in text_str.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            # Search for (A), (B)... at the end of the last line
            choice_match_end = re.search(r"\(?([A-Ea-e])\)?\.?\s*$", last_line)
            if choice_match_end:
                return choice_match_end.group(1).upper()
            
            # Search for just the letter if it's the only thing on the last line (e.g. "A")
            if len(last_line) == 1 and last_line.upper() in "ABCDE":
                return last_line.upper()

        # Fallback: Find the last occurrence of (A), (B), ... or A, B, ... if it's a single letter.
        # This is more risky as it might pick up choices mentioned in the reasoning.
        # We'll look for it near the end of the text.
        last_part_text = text_str[-50:] # Look in the last 50 characters
        choices_found = re.findall(r"\(([A-Ea-e])\)|^\s*([A-Ea-e])\s*$", last_part_text, re.MULTILINE) # (X) or X on its own line
        if choices_found:
            # choices_found will be a list of tuples, e.g., [('A', ''), ('', 'B')]
            # We need to get the actual letter from the tuple
            last_choice = None
            for choice_tuple in reversed(choices_found):
                actual_letter = next((c for c in choice_tuple if c), None)
                if actual_letter:
                    last_choice = actual_letter.upper()
                    break
            if last_choice:
                return last_choice
        
        return None

    def _extract_answer(self, text: str) -> Optional[Union[float, str, bool]]:
        """
        Dispatcher for answer extraction based on dataset_name.
        """
        if self.dataset_name == "gsm8k":
            return self._extract_answer_gsm8k(text)
        elif self.dataset_name == "strategyqa":
            return self._extract_answer_strategyqa(text)
        elif self.dataset_name == "math":
            return self._extract_answer_math(text)
        elif self.dataset_name == "commonsenseqa":
            return self._extract_answer_commonsenseqa(text)
        else:
            # Default or fallback if dataset_name is not set or not recognized
            # For now, let's assume GSM8K-like numeric extraction as a default
            # if no specific dataset logic is found. This might need adjustment.
            print(f"Warning: No specific answer extractor for dataset '{self.dataset_name}'. Falling back to GSM8K-like numeric extraction.")
            return self._extract_answer_gsm8k(text)

    def parse_output(self, model_output: str, **kwargs) -> Dict[str, Any]:
        """
        Parses the model output string and extracts the answer.
        (This is the method for single outputs, used by ZeroShot, FewShot etc.)
        """
        answer = self._extract_answer(model_output)
        if answer is not None:
            return {"answer": answer, "model_output_full": model_output}
        else:
            return {"answer": None, "model_output_full": model_output, "parsing_error": "Could not extract answer from model output"}


class ZeroShotPrompt(PromptTemplate):
    """Zero-shot prompt template."""
    
    def __init__(self, dataset_name: str, system_prompt: Optional[str] = None):
        super().__init__("zero_shot", system_prompt=system_prompt, dataset_name=dataset_name)
    
    def format(self, question: str, examples: List[Dict[str, str]] = None) -> str:
        # For CommonsenseQA, the question itself should contain the choices.
        # The load_dataset function for commonsenseqa should prepare the question string accordingly.
        instruction = "Solve the following problem:"
        if self.dataset_name == "commonsenseqa":
            instruction = "Answer the following multiple-choice question. Provide the letter of the correct answer."
        elif self.dataset_name == "math":
            instruction = "Solve the following math problem. Show your work and provide the final answer in a \\boxed{}."
        elif self.dataset_name == "strategyqa":
            instruction = "Answer the following question with Yes or No, and provide your reasoning."


        return f"""{instruction}
{question}

Make sure to show your work step-by-step (if applicable), and finish with the final answer clearly stated.
For multiple choice, the final answer should be the letter of the correct option (e.g., "The final answer is (A)").
For math problems, the final answer should be enclosed in \\boxed{{}} (e.g., "The final answer is \\boxed{{solution}}").
For Yes/No questions, the final answer should be "Yes" or "No" (e.g., "The final answer is Yes").
For other problems, state "The final answer is [your answer]".
"""
    
    def parse_output(self, model_output: str, **kwargs) -> Dict[str, Any]:
        reasoning_text = model_output 
        answer_val = None
        parsing_error = None

        # Try specific pattern first, as instructed by the prompt:
        # "Therefore, the answer is: [your answer]"
        # This regex looks for variations of that phrase.
        
        # Adjust pattern for StrategyQA to capture Yes/No
        if self.dataset_name == "strategyqa":
            specific_pattern = r"(?:Therefore,\s*the\s*answer\s*is|The\s*final\s*answer\s*is)\s*[:\s]*(Yes|No|True|False)\b"
            match = re.search(specific_pattern, model_output, re.IGNORECASE)
            if match:
                answer_str = match.group(1).lower()
                if answer_str in ["yes", "true"]:
                    answer_val = True
                elif answer_str in ["no", "false"]:
                    answer_val = False
                else:
                    parsing_error = "Found specific answer phrase for StrategyQA but value was not Yes/No/True/False."
            
        elif self.dataset_name == "gsm8k": # Original numeric pattern
            specific_pattern = r"(?:Therefore,\s*the\s*answer\s*is|The\s*final\s*answer\s*is)\s*[:\s]*([-+]?\d*\.?\d+)"
            match = re.search(specific_pattern, model_output, re.IGNORECASE)
            if match:
                try:
                    answer_val = float(match.group(1).replace(",", ""))
                except ValueError:
                    parsing_error = "Found specific answer phrase but number parsing failed."
        else: # Fallback for other datasets or if dataset_name is None
            # Use the generic _extract_answer which will dispatch
            pass # Will be handled by the fallback below

        if answer_val is None: 
            # If specific pattern failed or wasn't found for the current dataset,
            # fallback to the robust base dispatcher _extract_answer
            answer_val = self._extract_answer(model_output) # Uses the dispatcher
            if answer_val is None and not parsing_error: 
                parsing_error = f"Could not extract answer using dispatched _extract_answer for dataset {self.dataset_name}."
            elif answer_val is not None: # If dispatched extractor succeeded
                parsing_error = None 
        
        final_parsing_error = parsing_error if answer_val is None else None

        return {"answer": answer_val, 
                "reasoning": reasoning_text, 
                "model_output_full": model_output,
                "parsing_error": final_parsing_error}


class FewShotPrompt(PromptTemplate):
    """Few-shot prompt template with type-matched example selection for MATH."""
    
    def __init__(self, dataset_name: str, train_dataset: Optional[List[Dict[str, Any]]] = None, system_prompt: Optional[str] = None, num_examples: int = 3):
        super().__init__("few_shot", system_prompt=system_prompt, dataset_name=dataset_name)
        self.num_examples = num_examples
        self.train_dataset = train_dataset if train_dataset else []
        
        if self.dataset_name == "math" and not self.train_dataset:
             print(f"Warning: FewShotPrompt for MATH dataset initialized without a training dataset. Type matching disabled.")
        elif not self.train_dataset and self.dataset_name not in ALL_FEW_SHOT_EXAMPLES:
             print(f"Warning: No training data or hardcoded examples for dataset '{self.dataset_name}'. FewShotPrompt might act like ZeroShotPrompt.")

    def _get_examples_for_datapoint(self, data_point: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Selects few-shot examples, using type matching for MATH dataset."""
        
        # If not MATH dataset, or no train data provided, use hardcoded examples if available
        if self.dataset_name != "math" or not self.train_dataset:
            if self.dataset_name in ALL_FEW_SHOT_EXAMPLES:
                # Use hardcoded examples, ensuring not to select the data_point itself if it happens to be in the list
                hardcoded = ALL_FEW_SHOT_EXAMPLES[self.dataset_name]
                # Basic check to prevent using the exact same question if possible
                available_hardcoded = [ex for ex in hardcoded if ex.get("question") != data_point.get("question")]
                return random.sample(available_hardcoded, min(self.num_examples, len(available_hardcoded)))
            else:
                return [] # No examples available

        # MATH dataset: Type-matched selection
        target_type = data_point.get("type")
        if not target_type:
            print(f"Warning: Target data point missing 'type' field. Falling back to random selection for MATH.")
            # Exclude the current data_point from the pool before random sampling
            pool = [ex for ex in self.train_dataset if ex.get("id") != data_point.get("id")]
            return random.sample(pool, min(self.num_examples, len(pool)))

        # Filter training data for matching type, excluding the data_point itself
        type_matched_examples = [
            ex for ex in self.train_dataset 
            if ex.get("type") == target_type and ex.get("id") != data_point.get("id")
        ]
        
        selected_examples = []
        if len(type_matched_examples) >= self.num_examples:
            # Enough type-matched examples found
            selected_examples = random.sample(type_matched_examples, self.num_examples)
        else:
            # Not enough type-matched examples, use all found and fill with random
            selected_examples.extend(type_matched_examples)
            num_needed = self.num_examples - len(selected_examples)
            
            # Pool of other examples (different type or no type), excluding self and already selected
            selected_ids = {ex.get("id") for ex in selected_examples} | {data_point.get("id")}
            other_examples = [
                ex for ex in self.train_dataset 
                if ex.get("id") not in selected_ids
            ]
            
            if len(other_examples) >= num_needed:
                selected_examples.extend(random.sample(other_examples, num_needed))
            else:
                # Still not enough, use all available other examples
                selected_examples.extend(other_examples)
                if len(selected_examples) < self.num_examples:
                    print(f"Warning: Could only find {len(selected_examples)} examples in total for MATH type '{target_type}' (needed {self.num_examples}).")

        return selected_examples

    def format(self, data_point: Dict[str, Any], examples: Optional[List[Dict[str, str]]] = None) -> str:
        # 'examples' argument is ignored; selection logic is in _get_examples_for_datapoint
        question = data_point.get("question", "")
        selected_examples = self._get_examples_for_datapoint(data_point)

        prompt_parts = []
        if self.system_prompt:
            prompt_parts.append(self.system_prompt)

        instruction = "Solve the following problems. For each problem, provide a step-by-step explanation and then the final answer."
        if self.dataset_name == "commonsenseqa":
            instruction = "Answer the following multiple-choice questions. For each question, provide your reasoning and then the letter of the correct answer."
        elif self.dataset_name == "math":
            instruction = "Solve the following math problems. For each problem, show your work and provide the final answer in a \\\\boxed{}."
        elif self.dataset_name == "strategyqa":
            instruction = "Answer the following questions with Yes or No, and provide your reasoning."
        
        prompt_parts.append(instruction)
        prompt_parts.append("\\n--- Examples ---")

        if not selected_examples:
            print(f"Warning: No examples selected for FewShotPrompt for dataset {self.dataset_name}")
        
        for ex in selected_examples:
            # Determine the correct key for the answer/explanation
            # Use 'answer' field from loaded MATH data which contains the boxed solution.
            # Adapt logic for other datasets if necessary.
            answer_explanation_text = ""
            if self.dataset_name == "math":
                # Assuming loaded MATH data has 'problem' and 'answer' (which is the solution/boxed answer)
                # We need to reconstruct the "Explanation and Answer" format.
                # If 'solution' field is available, prefer that.
                solution = ex.get("solution") # Check if full solution is loaded
                answer = ex.get("answer")     # Check for just the boxed answer
                if solution:
                    answer_explanation_text = solution # Use the full solution if available
                elif answer:
                     # Fallback: Construct a basic explanation if only answer is present
                     answer_explanation_text = f"[Explanation leading to the answer...]\\nThe final answer is \\\\boxed{{{answer}}}"
                else:
                    answer_explanation_text = "No explanation or answer available."

            elif self.dataset_name == "gsm8k":
                 # Use the structure from hardcoded examples
                 answer_explanation_text = ex.get("answer_explanation", "No explanation available.")
                 # Ensure the key matches how GSM8K examples are loaded or defined.
                 # If loaded dynamically, might need keys 'question', 'answer'. Check loader.
                 # Example construction if only 'answer' is available:
                 # answer_explanation_text = f"[Explanation...]\\nThe final answer is #### {ex.get('answer', '')}"
            
            elif self.dataset_name == "strategyqa":
                # Use the structure from hardcoded examples
                answer_explanation_text = ex.get("answer_explanation", "No explanation available.")
                # Example construction if only 'answer' (yes/no) is available:
                # answer_explanation_text = f"[Reasoning...]\\nThe final answer is {ex.get('answer', 'Yes/No?')}."

            elif self.dataset_name == "commonsenseqa":
                 # Use the structure from hardcoded examples
                 answer_explanation_text = ex.get("answer_explanation", "No explanation available.")
                 # Example construction if only 'answer' (A/B/C..) is available:
                 # answer_explanation_text = f"[Reasoning...]\\nThe final answer is ({ex.get('answer', '?')})."
            
            else: # Default fallback
                 answer_explanation_text = ex.get("answer_explanation") or ex.get("solution") or ex.get("answer", "No explanation or answer available.")


            prompt_parts.append(f"\\nQuestion: {ex.get('question', 'No question provided.')}")
            # Use the determined answer/explanation text
            prompt_parts.append(f"Explanation and Answer:\\n{answer_explanation_text}") 


        prompt_parts.append("\\n--- Problem to Solve ---")
        prompt_parts.append(f"\\nQuestion: {question}")
        prompt_parts.append("Explanation and Answer:") # Added this line to prompt the model correctly
        
        return "\n".join(prompt_parts)
    
    # No parse_output override needed, uses base class method.


class ChainOfThoughtPrompt(PromptTemplate):
    """Chain-of-Thought prompt template."""
    
    def __init__(self, dataset_name: str, system_prompt: Optional[str] = None):
        super().__init__("cot", system_prompt=system_prompt, dataset_name=dataset_name)
    
    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        if self.dataset_name == "strategyqa":
            return f"{task_input}\nLet's think step by step. Your final answer must be Yes or No."
        return f"{task_input}\nLet's think step by step."

# Make sure SelfConsistencyPrompt class is defined here
class SelfConsistencyPrompt(PromptTemplate):
    """Self-Consistency prompt template (extends CoT)."""
    
    def __init__(self, dataset_name: str, system_prompt: Optional[str] = None, num_samples: int = 5):
        super().__init__("self_consistency", system_prompt=system_prompt, dataset_name=dataset_name)
        self.num_samples = num_samples

    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        """Formats the prompt for generating one sample in self-consistency."""
        # Self-consistency often builds on CoT, so the format is similar.
        if self.dataset_name == "strategyqa":
            return f"{task_input}\nLet's think step by step. Your final answer must be Yes or No."
        # General CoT-like instruction for other datasets or if dataset_name is not strategyqa
        return f"{task_input}\nLet's think step by step."

    def parse_output(self, model_outputs: List[str], question: Optional[str] = None) -> Dict[str, Any]:
        parsed_answers = []
        full_reasoning_for_majority_vote = "" 
        all_reasonings = [] 

        for output_str in model_outputs:
            all_reasonings.append(output_str)
            answer = self._extract_answer(output_str)
            if answer is not None:
                parsed_answers.append(answer)

        if not parsed_answers:
            return {
                "answer": None, 
                "reasoning": "Self-consistency: No valid answers found in samples.", 
                "sample_outputs": model_outputs, 
                "model_output_full": "".join(model_outputs if isinstance(model_outputs, list) else [str(model_outputs)])
            }

        answer_counts = Counter(parsed_answers)
        most_common_answer = answer_counts.most_common(1)[0][0]
        
        # Find one of the outputs that resulted in the most_common_answer to use its reasoning
        # This is a heuristic, could be improved (e.g. shortest, longest, first)
        # Ensure parsed_answers has elements and i is a valid index
        if parsed_answers:
            for i, output_str in enumerate(model_outputs):
                if i < len(parsed_answers) and parsed_answers[i] == most_common_answer:
                    full_reasoning_for_majority_vote = output_str
                    break
        
        final_answer = most_common_answer

        return {
            "answer": final_answer,
            "reasoning": f"Self-consistency: Chose '{final_answer}' from {len(parsed_answers)} valid samples. Reasoning from one such sample: {full_reasoning_for_majority_vote}",
            "model_output_full": full_reasoning_for_majority_vote, 
            "sample_outputs": model_outputs,
            "parsed_sample_answers": parsed_answers 
        }

class SelfReflectionPrompt(PromptTemplate):
    """Self-Reflection prompt template."""
    
    def __init__(self, dataset_name: str, system_prompt: Optional[str] = None, max_reflections: int = 3):
        super().__init__("self_reflection", system_prompt=system_prompt, dataset_name=dataset_name)
        self.max_reflections = max_reflections
        # Initial prompt is typically CoT-like
        self.initial_prompt_template = f"Question: {{question}}\nLet's think step by step to reach the solution."
        # Reflection prompt asks to critique and improve
        self.reflection_prompt_template = (
            "You are a helpful and meticulous AI assistant. "
            "You will be given a math or reasoning problem, a proposed solution, and a critique of that solution. "
            "Your task is to reflect on the critique and provide an improved, step-by-step solution. "
            "If the critique points out an error, correct it. If the critique asks for clarification, provide it. "
            "Ensure your final answer is clearly marked, for example, using 'The final answer is ...'.\n\n"
            "Problem: {question}\n"
            "Proposed Solution Attempt:\n{initial_solution}\n"
            "Critique of Proposed Solution:\n{critique}\n\n"
            "Improved Step-by-Step Solution:"
        )
        # Self-critique prompt (model critiques its own answer)
        # This might need to be dataset-specific if the nature of "critique" changes.
        self.self_critique_prompt_template = (
            "You are a self-critical AI assistant. You will be given a question and your own proposed solution. "
            "Analyze your solution step-by-step. Are there any logical fallacies, calculation errors, or missed steps? "
            "Is the reasoning clear? Is the final answer correct and well-supported? "
            "Provide a concise critique. If the solution is perfect, state that. "
            "If there are errors, point them out specifically.\n\n"
            "Question: {question}\n"
            "Your Proposed Solution:\n{solution_to_critique}\n\n"
            "Your Critique:"
        )

    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        """Formats the initial prompt for the self-reflection process."""
        # The 'examples' argument is not used by the initial prompt in self-reflection by default.
        return self.format_initial_prompt(task_input)

    def format_initial_prompt(self, task_input: str) -> str:
        return self.initial_prompt_template.format(question=task_input)

    def format_reflection_prompt(self, task_input: str, initial_solution: str, critique: str) -> str:
        return self.reflection_prompt_template.format(question=task_input, initial_solution=initial_solution, critique=critique)

    def format_self_critique_prompt(self, task_input: str, solution_to_critique: str) -> str:
        return self.self_critique_prompt_template.format(question=task_input, solution_to_critique=solution_to_critique)

    def parse_output(self, model_output: str, reflection_model_output: Optional[str] = None, question: Optional[str] = None) -> Dict[str, Any]:
        """
        Parses the output of a self-reflection process.
        If only model_output is given, it's the first pass.
        If reflection_model_output is also given, it's the output after reflection.
        """
        # Parse the initial output
        initial_answer = self._extract_answer(model_output)
        
        if reflection_model_output is None:
            # This is the parse of the first pass (before any reflection)
            return {
                "answer": initial_answer, # This is the 'initial_answer_from_model'
                "reasoning": model_output,
                "model_output_full": model_output, # Added model_output_full
                "parsing_error": "Initial pass, no reflection output yet." if initial_answer is None else None
            }
        else:
            # This is the parse of the output *after* reflection
            final_answer_from_reflection = self._extract_answer(reflection_model_output)
            return {
                "initial_answer_from_model": initial_answer, # Answer from the first pass
                "initial_reasoning": model_output,
                "reflection_reasoning": reflection_model_output, # The text of the improved solution
                "final_answer_from_reflection": final_answer_from_reflection, # Answer from the improved solution
                # The primary 'answer' for evaluation should be final_answer_from_reflection
                "answer": final_answer_from_reflection, 
                "model_output_full": reflection_model_output, # Added model_output_full
                "parsing_error": "Could not extract final answer after reflection." if final_answer_from_reflection is None else None
            }


class ReActPrompt(PromptTemplate):
    """ReAct prompt template (Reasoning + Acting)."""
    
    # Define a default system prompt for ReAct
    REACT_SYSTEM_PROMPT = (
        "You are a reasoning agent. You need to solve a given problem by breaking it down into steps. "
        "At each step, you can either think about the problem (Thought), perform a calculation (Action: Compute[expression]), "
        "or provide the final answer (Answer: [your answer]). "
        "Ensure your calculations are accurate and your reasoning is clear."
    )
    
    def __init__(self, dataset_name: str, system_prompt: Optional[str] = None):
        # Use the class attribute REACT_SYSTEM_PROMPT if no specific system_prompt is provided
        super().__init__("react", system_prompt=system_prompt if system_prompt else ReActPrompt.REACT_SYSTEM_PROMPT, dataset_name=dataset_name)
    
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
    def parse_output(self, model_output: str, question: Optional[str] = None) -> Dict[str, Any]: # Added question param for consistency
        lines = model_output.strip().split('\n')
        thoughts = []
        actions = []
        parsed_answer_val = None 
        parsing_error_final = None
        
        # Try to find "Answer:" line first
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("Thought:"):
                thoughts.append(line_stripped[len("Thought:"):].strip())
            elif line_stripped.startswith("Action: Compute["): # Assuming Compute for now
                expr = line_stripped[len("Action: Compute["):]
                expr = expr.split("]")[0] if "]" in expr else expr
                actions.append(expr)
            elif line_stripped.startswith("Answer:"):
                answer_text_after_marker = line_stripped[len("Answer:"):].strip()
                # Use dataset-aware _extract_answer
                parsed_val_from_marker = self._extract_answer(answer_text_after_marker)
                if parsed_val_from_marker is not None:
                    parsed_answer_val = parsed_val_from_marker
                    parsing_error_final = None 
                    break 
                else:
                    if parsing_error_final is None:
                         parsing_error_final = f"Failed to parse content after 'Answer:' marker using {self.dataset_name} extractor."
        
        if parsed_answer_val is None:
            # Fallback to parsing the whole output using dataset-aware _extract_answer
            parsed_val_from_full = self._extract_answer(model_output)
            if parsed_val_from_full is not None:
                parsed_answer_val = parsed_val_from_full
                parsing_error_final = None # Succeeded with full parse
            elif parsing_error_final is None: # Only set error if not already set by marker parsing
                parsing_error_final = f"Failed to parse whole output using {self.dataset_name} extractor."


        result = {
            "model_output_full": model_output, 
            "thoughts": thoughts,
            "actions": actions,
            "answer": parsed_answer_val,
            "parsing_error": parsing_error_final if parsed_answer_val is None else None
        }
        return result

    def _extract_finish_answer(self, last_action_str: str) -> Optional[Any]:
        # This method seems specific to an older ReAct style.
        # The current parse_output tries to find "Answer:" or parses the whole output.
        # If you intend to use "finish[...]" as the final action, parse_output needs to look for that.
        # For now, it uses self._extract_answer on the content if "finish[]" is found.
        if last_action_str and last_action_str.lower().startswith("finish[") and last_action_str.endswith("]"):
            answer_content = last_action_str[len("finish["):-1].strip()
            return self._extract_answer(answer_content) # Uses dataset-aware extraction
        return None


# Dictionary mapping prompt types to their classes
PROMPT_TEMPLATE_CLASSES = { # Renamed from PROMPT_TEMPLATES
    "zero_shot": ZeroShotPrompt,
    "few_shot": FewShotPrompt,
    "cot": ChainOfThoughtPrompt,
    "self_consistency": SelfConsistencyPrompt,
    "self_reflection": SelfReflectionPrompt,
    "react": ReActPrompt
}


def get_prompt_template(template_name: str, dataset_name: str, **kwargs) -> PromptTemplate:
    """Get a prompt template by type, ensuring dataset_name is passed."""
    if template_name not in PROMPT_TEMPLATE_CLASSES: # Use new name
        raise ValueError(f"Unknown prompt template name: {template_name}")
    
    # Pass dataset_name and any other kwargs to the constructor of the prompt template class
    # Ensure train_dataset is passed specifically to FewShotPrompt if needed
    template_class = PROMPT_TEMPLATE_CLASSES[template_name]
    
    # Special handling for FewShotPrompt to pass train_dataset if provided in kwargs
    if template_name == "few_shot" and "train_dataset" in kwargs:
         # Extract train_dataset and pass it explicitly
         train_data = kwargs.pop("train_dataset")
         return template_class(dataset_name=dataset_name, train_dataset=train_data, **kwargs)
    else:
         # For other templates or if train_dataset not provided for few-shot
         # Remove train_dataset from kwargs if present but not needed by the class
         kwargs.pop("train_dataset", None) 
         return template_class(dataset_name=dataset_name, **kwargs) 