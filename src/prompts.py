"""
Prompt templates for different reasoning strategies.
"""
from typing import List, Dict, Any, Optional

class PromptTemplate:
    """Base class for all prompt templates."""
    
    def __init__(self, name: str):
        self.name = name
    
    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        """Format the prompt template with the task input."""
        raise NotImplementedError
    
    def parse_output(self, output: str) -> Dict[str, Any]:
        """Parse the model output."""
        # Default implementation just returns the raw output
        return {"raw_output": output, "answer": output.strip()}


class ZeroShotPrompt(PromptTemplate):
    """Zero-shot prompt template."""
    
    def __init__(self):
        super().__init__("zero_shot")
    
    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        return task_input


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


class ChainOfThoughtPrompt(PromptTemplate):
    """Chain-of-Thought prompt template."""
    
    def __init__(self):
        super().__init__("cot")
    
    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        return f"{task_input}\nLet's think step by step."
    
    def parse_output(self, output: str) -> Dict[str, Any]:
        # Try to extract the final answer from the reasoning chain
        lines = output.strip().split('\n')
        answer = lines[-1].strip()
        reasoning = '\n'.join(lines[:-1]) if len(lines) > 1 else ""
        
        return {
            "raw_output": output,
            "reasoning": reasoning,
            "answer": answer,
            "reasoning_length": len(reasoning.split())
        }


class SelfConsistencyPrompt(ChainOfThoughtPrompt):
    """Self-Consistency prompt template (extends CoT)."""
    
    def __init__(self, num_samples: int = 5):
        super().__init__()
        self.name = "self_consistency"
        self.num_samples = num_samples


class SelfReflectionPrompt(PromptTemplate):
    """Self-Reflection prompt template."""
    
    def __init__(self):
        super().__init__("self_reflection")
    
    def format(self, task_input: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        cot_prompt = ChainOfThoughtPrompt().format(task_input, examples)
        return cot_prompt
    
    def format_reflection(self, initial_output: str) -> str:
        """Format the reflection prompt based on initial output."""
        return f"{initial_output}\n\nIs your answer correct? Why?"
    
    def parse_output(self, output: str, reflection_output: str) -> Dict[str, Any]:
        # Parse both initial output and reflection
        cot_result = ChainOfThoughtPrompt().parse_output(output)
        
        # Extract final answer potentially revised after reflection
        lines = reflection_output.strip().split('\n')
        final_answer = lines[-1].strip()
        
        return {
            "raw_output": output + "\n\n" + reflection_output,
            "initial_reasoning": cot_result["reasoning"],
            "initial_answer": cot_result["answer"],
            "reflection": reflection_output,
            "final_answer": final_answer
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
        final_answer = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thoughts.append(line[len("Thought:"):].strip())
            elif line.startswith("Action: Compute["):
                # Extract the computation
                expr = line[len("Action: Compute["):]
                expr = expr.split("]")[0] if "]" in expr else expr
                actions.append(expr)
            elif line.startswith("Answer:"):
                final_answer = line[len("Answer:"):].strip()
        
        return {
            "raw_output": output,
            "thoughts": thoughts,
            "actions": actions,
            "answer": final_answer
        }


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