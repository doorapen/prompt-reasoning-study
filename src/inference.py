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
import requests
import json

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
    """Interface for DeepSeek via API."""
    
    _debug_printed_once = False
    
    def __init__(self):
        super().__init__("deepseek-r1")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
        self.api_base = "https://api.deepseek.com"  # Check the actual endpoint from DeepSeek docs
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate text from DeepSeek API."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Adjust temperature: Use provided temperature. If it's 0, use a small default.
            # This allows higher temperatures for CoT, self-consistency etc. if specified by run_inference
            effective_temperature = temperature
            if effective_temperature <= 0.01: # If very low or zero, use a small default for determinism
                effective_temperature = 0.1 # Or 0.0, but some models require > 0
            elif effective_temperature > 1.0: # Cap temperature if it's too high
                effective_temperature = 1.0


            # Use a model that's better for mathematical reasoning if available
            data = {
                "model": "deepseek-coder",  # Try their coder model which may be better at math
                "messages": [{"role": "user", "content": prompt}],
                "temperature": effective_temperature, # Use the adjusted temperature
                "max_tokens": max_tokens
            }
            
            # Define a timeout (e.g., 60 seconds for connect and read)
            timeout_seconds = 60 

            response = requests.post(
                f"{self.api_base}/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=timeout_seconds # Add timeout here
            )
            
            if response.status_code != 200:
                print(f"Error from DeepSeek API: {response.status_code}, {response.text}")
                return f"ERROR: API returned status code {response.status_code}"
            
            response_json = response.json()
            
            # Debug only the first time
            if not DeepSeekInterface._debug_printed_once:
                print(f"\nDEBUG: DeepSeek API response structure: {list(response_json.keys())}")
                DeepSeekInterface._debug_printed_once = True
            
            if "choices" in response_json and len(response_json["choices"]) > 0:
                if "message" in response_json["choices"][0]:
                    return response_json["choices"][0]["message"]["content"]
                else:
                    print(f"Unexpected response structure: {response_json['choices'][0]}")
                    return response_json["choices"][0].get("text", "ERROR: Unknown response format")
            else:
                print(f"No choices in response: {response_json}")
                return "ERROR: No valid response from API"
        
        except requests.exceptions.Timeout:
            print(f"Timeout error connecting to DeepSeek API for prompt: {prompt[:100]}...")
            return "ERROR: API request timed out"
        except requests.exceptions.RequestException as e: # Catch other requests-related errors
            print(f"Request error during DeepSeek API generation: {e}")
            return f"ERROR: API request failed - {str(e)}"
        except Exception as e:
            print(f"Error in DeepSeek API generation: {e}")
            return f"ERROR: {str(e)}"


def get_model_interface(model_name: str) -> ModelInterface:
    """
    Get the appropriate model interface based on the model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelInterface instance
    """
    if model_name == "gpt-4":
        return GPT4Interface()
    elif model_name == "deepseek-r1":
        # Don't pass the old model ID here - let it use the default we set in the class
        return DeepSeekInterface()
    else:
        raise ValueError(f"Unknown model: {model_name}")


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
                        temperature=max(0.7, temperature),  # Often uses higher temperature
                        max_tokens=max_tokens
                    )
                    outputs.append(output)
                
                # Process through voting or other aggregation
                from src.decoding import majority_vote
                final_output_text = majority_vote(outputs, prompt_template) # This is the text from majority vote

                # Parse the final_output_text to get the 'answer' field and other potential fields
                parsed_final_answer_details = prompt_template.parse_output(final_output_text)

                parsed_result = {
                    "sample_outputs": outputs,
                    "majority_vote_text": final_output_text, # Store the text from majority vote
                    "answer": parsed_final_answer_details.get("answer"), # Get the parsed answer
                    "model_output_full": final_output_text, # The full output for this method is the majority vote result
                    # You might want to add other fields from parsed_final_answer_details if needed
                }
                # Ensure 'answer' is present, even if None, for consistent structure
                if "answer" not in parsed_result:
                    parsed_result["answer"] = None
            
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
                
                # Ensure answer field is set for evaluation
                if "final_answer" in parsed_result and not parsed_result.get("answer"):
                    parsed_result["answer"] = parsed_result["final_answer"]
            
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