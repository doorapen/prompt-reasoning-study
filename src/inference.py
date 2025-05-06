"""
Model inference utilities for GPT-4 and DeepSeek-R1.
"""
import os
import time
from typing import List, Dict, Any, Optional, Union, Callable

import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from src.prompts import PromptTemplate


class ModelInterface:
    """Base class for all LLM interfaces."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate(self, prompt: str, 
                 temperature: float = 0.0, 
                 max_tokens: int = 1024) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError


class GPT4Interface(ModelInterface):
    """Interface for OpenAI's GPT-4 model."""
    
    def __init__(self, model_version: str = "gpt-4o", api_key: str = None):
        super().__init__(model_version)
        # Use provided API key or try to get from environment
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please provide it explicitly or set the OPENAI_API_KEY environment variable.")
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, 
                 temperature: float = 0.0, 
                 max_tokens: int = 1024) -> str:
        """Generate text from GPT-4."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in GPT-4 generation: {e}")
            # Add a retry mechanism if needed
            time.sleep(2)  # Backoff before retry
            return f"ERROR: {str(e)}"


class DeepSeekInterface(ModelInterface):
    """Interface for DeepSeek-R1 model."""
    
    def __init__(self, model_id: str = "deepseek-ai/deepseek-r1-7b"):
        super().__init__(model_id)
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config
        )
    
    def generate(self, prompt: str, 
                 temperature: float = 0.0, 
                 max_tokens: int = 1024) -> str:
        """Generate text from DeepSeek-R1."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 1e-7),  # Avoid exact 0
                    do_sample=temperature > 0
                )
            
            # Extract the generated text (excluding the prompt)
            prompt_length = inputs.input_ids.shape[1]
            generated_ids = outputs[0][prompt_length:]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        except Exception as e:
            print(f"Error in DeepSeek generation: {e}")
            return f"ERROR: {str(e)}"


def get_model_interface(model_type: str) -> ModelInterface:
    """Get a model interface by type."""
    if model_type.lower() == "gpt-4":
        return GPT4Interface()
    elif model_type.lower() == "deepseek-r1":
        return DeepSeekInterface()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_inference(
    model: ModelInterface,
    prompt_template: PromptTemplate,
    dataset: List[Dict[str, Any]],
    examples: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    batch_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Run inference on a dataset using the given model and prompt template.
    
    Args:
        model: The model interface
        prompt_template: The prompt template
        dataset: List of data points to process
        examples: Optional examples for few-shot prompting
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        batch_callback: Optional callback to process results in batches
        batch_size: Size of batches for processing and callbacks
        
    Returns:
        List of results with model outputs
    """
    results = []
    current_batch = []
    
    for i, data_point in enumerate(tqdm(dataset, desc=f"Running {model.model_name} with {prompt_template.name}")):
        try:
            # Format the prompt
            prompt = prompt_template.format(data_point["question"], examples)
            
            # Special handling for self-consistency
            if prompt_template.name == "self_consistency":
                # Run multiple times and collect results
                outputs = []
                for _ in range(prompt_template.num_samples):
                    output = model.generate(
                        prompt, 
                        temperature=max(0.7, temperature),  # Use higher temperature for diversity
                        max_tokens=max_tokens
                    )
                    outputs.append(output)
                
                # Process through voting or other aggregation
                from src.decoding import majority_vote
                final_output = majority_vote(outputs, prompt_template)
                parsed_result = {
                    "sample_outputs": outputs,
                    "majority_vote": final_output,
                    **prompt_template.parse_output(outputs[0])  # Parse first sample for analysis
                }
            
            # Special handling for self-reflection
            elif prompt_template.name == "self_reflection":
                # First generate the initial output
                initial_output = model.generate(prompt, temperature=temperature, max_tokens=max_tokens)
                
                # Then generate the reflection
                reflection_prompt = prompt_template.format_reflection(initial_output)
                reflection_output = model.generate(
                    reflection_prompt, 
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                parsed_result = prompt_template.parse_output(initial_output, reflection_output)
            
            # Standard handling for other prompt types
            else:
                output = model.generate(prompt, temperature=temperature, max_tokens=max_tokens)
                parsed_result = prompt_template.parse_output(output)
            
            # Combine the parsed result with the data point
            result = {
                "id": data_point.get("id", i),
                "question": data_point["question"],
                "true_answer": data_point.get("answer", ""),
                "prompt_type": prompt_template.name,
                "model": model.model_name,
                **parsed_result
            }
            
            results.append(result)
            current_batch.append(result)
            
            # Process batch if needed
            if batch_callback and len(current_batch) >= batch_size:
                batch_callback(current_batch)
                current_batch = []
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            results.append({
                "id": data_point.get("id", i),
                "question": data_point["question"],
                "error": str(e)
            })
    
    # Process any remaining items in the batch
    if batch_callback and current_batch:
        batch_callback(current_batch)
    
    return results 