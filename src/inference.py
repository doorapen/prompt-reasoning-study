"""
Model inference utilities for GPT-4 and DeepSeek-R1.
"""
import os
import time
from typing import List, Dict, Any, Optional, Union, Callable
import random # Added for jitter
import traceback # Add traceback import

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
        
        # Determine the API key to use.
        # Priority:
        # 1. api_key parameter passed to constructor.
        # 2. OPENAI_API_KEY environment variable.
        # 3. Prompt user if not found in the above.

        current_api_key = api_key # Use parameter if provided

        if not current_api_key:
            # Try to get API key from environment variable if not provided in constructor
            current_api_key = os.environ.get("OPENAI_API_KEY")
        
        if not current_api_key:
            # If still no key (not in constructor, not in env), prompt the user.
            # This block is adapted from the original code's fallback mechanism.
            import getpass # Import only when needed
            
            print("\n⚠️ OPENAI_API_KEY environment variable not found, and no API key provided to constructor.")
            current_api_key = getpass.getpass("Please enter your OpenAI API key: ")
            
            if not current_api_key: # Check if getpass returned an empty string
                raise ValueError("OpenAI API key is required and was not provided.") # Changed as per instruction
            
            # Set the API key in the environment for the current session if obtained via prompt.
            # This can be useful if other parts of the application (or library internals) 
            # might also try to read it from the environment.
            os.environ["OPENAI_API_KEY"] = current_api_key
            print("API key set for this session. For future runs, consider setting the OPENAI_API_KEY environment variable.")
        
        # Final check: ensure an API key was actually obtained and is not an empty string.
        if not current_api_key:
            # This case handles if api_key parameter was an empty string and other methods also failed
            # or returned an empty string that wasn't caught by the specific getpass check.
            raise ValueError("Failed to obtain a valid OpenAI API key. Ensure it's set via parameter, environment variable, or when prompted.")
            
        self.client = openai.OpenAI(api_key=current_api_key)
    
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
            # Prompt for API key if not found
            import getpass
            print("\n⚠️ DEEPSEEK_API_KEY environment variable not found.")
            self.api_key = getpass.getpass("Please enter your DeepSeek API key: ")
            if not self.api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable is not set and no key was provided when prompted.")
            # Temporarily set the environment variable for this session
            os.environ["DEEPSEEK_API_KEY"] = self.api_key
            print("DeepSeek API key set for this session. For future runs, consider setting the DEEPSEEK_API_KEY environment variable.")
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
            
            # --- Retry Logic ---
            max_retries = 3
            base_delay = 5 # seconds
            
            for attempt in range(max_retries):
                response = None # Initialize response to None for each attempt
                try:
                    response = requests.post(
                        f"{self.api_base}/v1/chat/completions",
                        headers=headers,
                        data=json.dumps(data),
                        timeout=timeout_seconds # Add timeout here
                    )
                    
                    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                    
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
                            # Return error instead of potentially partial text
                            return f"ERROR: Unexpected response structure {response_json['choices'][0].keys()}"
                    else:
                        print(f"No choices in response: {response_json}")
                        return "ERROR: No valid response from API"

                except requests.exceptions.Timeout as e:
                    print(f"Timeout error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt == max_retries - 1:
                        print(f"Max retries reached. Failing for prompt: {prompt[:100]}...")
                        return "ERROR: API request timed out after retries"
                except requests.exceptions.RequestException as e: 
                    # Catch other requests-related errors (like connection errors, status codes)
                    print(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
                    # Check if the error is likely retryable (e.g., 5xx server errors, connection errors)
                    # Don't retry on 4xx client errors usually. raise_for_status() handles this.
                    is_retryable = response.status_code >= 500 if response and hasattr(response, 'status_code') else True # Assume retryable if status code unknown or response is None

                    if attempt == max_retries - 1 or not is_retryable:
                        print(f"Max retries reached or error not retryable. Failing for prompt: {prompt[:100]}...")
                        return f"ERROR: API request failed after retries - {str(e)}"
                
                # If we got here, an error occurred and we should retry
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1) # Exponential backoff with jitter
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                
            # This part should ideally not be reached if retry logic is correct, 
            # but acts as a fallback.
            return "ERROR: Failed after multiple retries."

        except Exception as e: # Catch any other unexpected errors during setup/logic
            print(f"Unexpected error in DeepSeek API generation logic: {e}")
            return f"ERROR: Unexpected - {str(e)}"


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
            # Pass the full data_point for few_shot to allow type matching
            if prompt_template.name == "few_shot":
                prompt = prompt_template.format(data_point, examples=None) # examples are handled internally now
            else:
                # Other prompt types just need the question string
                prompt = prompt_template.format(data_point.get("question", ""), examples)
            
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
                # final_output_text = majority_vote(outputs, prompt_template) # This is the text from majority vote
                # The majority voting is handled inside SelfConsistencyPrompt.parse_output

                # Parse the list of outputs to get the 'answer' field and other potential fields
                parsed_final_answer_details = prompt_template.parse_output(outputs) # Pass the list of outputs

                parsed_result = {
                    "sample_outputs": outputs,
                    # "majority_vote_text": final_output_text, # This will be part of parsed_final_answer_details if needed
                    "answer": parsed_final_answer_details.get("answer"), # Get the parsed answer
                    "model_output_full": parsed_final_answer_details.get("model_output_full", ""), # Get the reasoning for the majority answer
                    # Add other fields from parsed_final_answer_details
                    "reasoning": parsed_final_answer_details.get("reasoning", ""),
                    "parsed_sample_answers": parsed_final_answer_details.get("parsed_sample_answers")
                }
                # Ensure 'answer' is present, even if None, for consistent structure
                if "answer" not in parsed_result:
                    parsed_result["answer"] = None
            
            # Special handling for self-reflection
            elif prompt_template.name == "self_reflection":
                # 1. First generate the initial output/solution
                # The `prompt` variable is already formatted by the general `format` call for SelfReflectionPrompt,
                # which calls `format_initial_prompt`.
                initial_solution_output = model.generate(prompt, temperature=temperature, max_tokens=max_tokens)
                
                # 2. Generate a self-critique of the initial solution
                # We need the original question (task_input) for the critique prompt
                task_input = data_point.get("question", "")
                critique_prompt = prompt_template.format_self_critique_prompt(task_input=task_input, solution_to_critique=initial_solution_output)
                critique_output = model.generate(critique_prompt, temperature=temperature, max_tokens=max_tokens) # Max tokens for critique might be smaller
                
                # 3. Generate the final refined solution based on the critique
                refined_solution_prompt = prompt_template.format_reflection_prompt(
                    task_input=task_input, 
                    initial_solution=initial_solution_output, 
                    critique=critique_output
                )
                refined_solution_output = model.generate(refined_solution_prompt, temperature=temperature, max_tokens=max_tokens)
                                
                # Parse the outputs
                # The parse_output for SelfReflectionPrompt expects initial_output and the *final* refined_output
                parsed_result = prompt_template.parse_output(initial_solution_output, refined_solution_output)
                
                # Add the critique to the results for inspection
                parsed_result["critique_output"] = critique_output
                
                # Ensure answer field is set for evaluation (already handled in SelfReflectionPrompt.parse_output)
                # if "final_answer_from_reflection" in parsed_result and not parsed_result.get("answer"):
                #    parsed_result["answer"] = parsed_result["final_answer_from_reflection"]
            
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
            
            # Add reasoning_length for relevant prompt types
            reasoning_text_for_length = "" # Default to empty string
            if prompt_template.name in ["zero_shot", "cot"]:
                reasoning_text_for_length = parsed_result.get("reasoning") or parsed_result.get("model_output_full") or ""
            elif prompt_template.name == "few_shot":
                reasoning_text_for_length = parsed_result.get("model_output_full") or ""
            elif prompt_template.name == "self_consistency":
                reasoning_text_for_length = parsed_result.get("reasoning") or "" 
            elif prompt_template.name == "self_reflection":
                reasoning_text_for_length = parsed_result.get("reflection_reasoning") or parsed_result.get("initial_reasoning") or ""
            elif prompt_template.name == "react":
                reasoning_text_for_length = parsed_result.get("model_output_full") or ""
            
            result["reasoning_length"] = len(reasoning_text_for_length)
            
            results.append(result)
            current_batch.append(result)
            
            # Process batch if needed
            if batch_callback and len(current_batch) >= batch_size:
                batch_callback(current_batch)
                current_batch = []
                
        except Exception as e:
            print(f"Error processing item {data_point.get('id', i)}:")
            traceback.print_exc() # Print the full traceback
            results.append({
                "id": data_point.get("id", i),
                "question": data_point.get("question", "N/A"), # Use get for safety
                "error": str(e),
                "prompt_type": prompt_template.name, # Add prompt type to error record
                "model": model.model_name,          # Add model name to error record
            })
    
    # Process any remaining items in the batch
    if batch_callback and current_batch:
        batch_callback(current_batch)
    
    return results 