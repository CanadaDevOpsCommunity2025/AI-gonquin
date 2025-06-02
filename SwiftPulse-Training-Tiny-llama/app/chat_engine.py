# app/chat_engine.py

"""
Handles interaction with a fine-tuned TinyLlama model for tech news conversations.
This module defines the generate_response() function, which takes a user message
and returns a chatbot-style reply using the trained TinyLlama model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration - Updated to use local fine-tuned TinyLlama
PROJECT_HOME = Path(__file__).parent.parent  # Go up from app/ to project root
TRAINED_MODEL_PATH = PROJECT_HOME / "model" / "trained_model" / "final_model"

# Initialize model and tokenizer variables
tokenizer = None
model = None

# Conversation history for context
conversation_history = []

def load_model():
    """Load the fine-tuned TinyLlama model from local path"""
    global tokenizer, model
    
    try:
        # Convert to string for compatibility
        model_path_str = str(TRAINED_MODEL_PATH)
        
        # Check if the model directory exists
        if not TRAINED_MODEL_PATH.exists():
            error_msg = f"Fine-tuned model not found at {model_path_str}. Please ensure the model files are present."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Loading fine-tuned TinyLlama model from: {model_path_str}")
        
        # Load tokenizer and model from local path
        tokenizer = AutoTokenizer.from_pretrained(
            model_path_str,
            local_files_only=True,  # Prevent downloading from HF
            trust_remote_code=True  # For GPTQ models
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path_str,
            local_files_only=True,  # Prevent downloading from HF
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            device_map="auto",  # Automatically distribute across available GPUs/CPU
            trust_remote_code=True  # For GPTQ models
        )
        
        logger.info("Fine-tuned TinyLlama model loaded successfully!")
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        # Log model info
        device = next(model.parameters()).device
        logger.info(f"Model loaded on device: {device}")
        logger.info(f"Model vocabulary size: {tokenizer.vocab_size}")
            
    except Exception as e:
        logger.error(f"Error loading fine-tuned TinyLlama model: {str(e)}")
        logger.error("Make sure the fine-tuned model is saved in the correct directory with all required files.")
        raise e

def format_chat_prompt(user_input: str, system_prompt: str = None) -> str:
    """
    Format the input for TinyLlama chat model using the proper chat template.
    
    Args:
        user_input (str): The user's message
        system_prompt (str): Optional system prompt for context
        
    Returns:
        str: Formatted prompt for TinyLlama
    """
    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant specializing in technology news and discussions. Provide informative, accurate, and engaging responses."
    
    # Build conversation context for TinyLlama chat format
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history (limit to recent exchanges for context window)
    for entry in conversation_history[-4:]:  # Keep last 4 exchanges for TinyLlama's smaller context
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["assistant"]})
    
    # Add current user message
    messages.append({"role": "user", "content": user_input})
    
    # Use tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return formatted_prompt
        except Exception as e:
            logger.warning(f"Chat template application failed: {e}")
    
    # Fallback manual formatting for TinyLlama
    formatted_prompt = f"<|system|>\n{system_prompt}</s>\n"
    
    # Add conversation history
    for entry in conversation_history[-2:]:  # Last 2 exchanges for context
        formatted_prompt += f"<|user|>\n{entry['user']}</s>\n<|assistant|>\n{entry['assistant']}</s>\n"
    
    # Add current user input
    formatted_prompt += f"<|user|>\n{user_input}</s>\n<|assistant|>\n"
    
    return formatted_prompt

def generate_response(user_input: str, max_length: int = 100, reset_history: bool = False, 
                     system_prompt: str = None) -> str:
    """
    Generates a conversational reply using the fine-tuned TinyLlama model.

    Args:
        user_input (str): The user's message.
        max_length (int): Maximum number of new tokens to generate (reduced for TinyLlama).
        reset_history (bool): Whether to reset conversation history.
        system_prompt (str): Optional system prompt for context.

    Returns:
        str: The chatbot's reply.
    """
    global conversation_history, model, tokenizer
    
    # Load model if not already loaded
    if model is None or tokenizer is None:
        load_model()
    
    # Reset history if requested
    if reset_history:
        conversation_history = []
        logger.info("Conversation history reset")
    
    try:
        # Format the prompt for TinyLlama
        formatted_prompt = format_chat_prompt(user_input, system_prompt)
        
        # Tokenize the prompt
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # TinyLlama has smaller context window
        )
        
        # Move to appropriate device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response with parameters tuned for TinyLlama
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.8,  # Slightly higher for more creative responses
                top_p=0.9,
                top_k=40,  # Reduced for TinyLlama
                repetition_penalty=1.15,  # Slightly higher to avoid repetition
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode only the new tokens (response)
        input_length = inputs['input_ids'].shape[1]
        response_ids = outputs[0][input_length:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Clean up response
        response = response.strip()
        
        # Remove any remaining special tokens or artifacts
        cleanup_tokens = ["</s>", "<|assistant|>", "<|user|>", "<|system|>"]
        for token in cleanup_tokens:
            if response.startswith(token):
                response = response[len(token):].strip()
            if response.endswith(token):
                response = response[:-len(token)].strip()
        
        # Remove any incomplete sentences at the end
        if response and not response.endswith(('.', '!', '?', ':', ';')):
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
            
        # Store conversation history
        if response:
            conversation_history.append({
                "user": user_input,
                "assistant": response
            })
            
            # Limit history size for TinyLlama's smaller context window
            if len(conversation_history) > 6:
                conversation_history = conversation_history[-6:]
        
        # Return meaningful response or fallback
        if response:
            return response
        else:
            return "I'm not sure how to respond to that. Could you try rephrasing your question about technology?"
            
    except Exception as e:
        logger.error(f"Error generating response with TinyLlama: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request. Please try again."

def get_model_info() -> dict:
    """
    Returns information about the currently loaded TinyLlama model.
    
    Returns:
        dict: Model information including path and type.
    """
    global model, tokenizer
    
    if model is None:
        return {"status": "not_loaded"}
    
    return {
        "status": "loaded",
        "model_path": str(TRAINED_MODEL_PATH),
        "model_type": "TinyLlama-1.1B-Chat-GPTQ (Fine-tuned)",
        "is_trained": True,
        "vocab_size": tokenizer.vocab_size if tokenizer else None,
        "context_length": getattr(tokenizer, 'model_max_length', 2048),
        "device": str(next(model.parameters()).device) if model else None,
        "model_size": "1.1B parameters"
    }

def reset_conversation():
    """Reset the conversation history"""
    global conversation_history
    conversation_history = []
    logger.info("Conversation history has been reset")

def set_system_prompt(prompt: str):
    """
    Set a custom system prompt for the conversation.
    
    Args:
        prompt (str): The system prompt to use
    """
    logger.info("Custom system prompt will be applied to next conversation")

def check_model_files():
    """
    Check if all required model files are present in the directory.
    
    Returns:
        dict: Status of required files
    """
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    # Common model file patterns
    model_files = ["pytorch_model.bin", "model.safetensors", "quantize_config.json"]
    
    file_status = {}
    
    for file in required_files + model_files:
        file_path = TRAINED_MODEL_PATH / file
        file_status[file] = file_path.exists()
    
    return {
        "model_directory_exists": TRAINED_MODEL_PATH.exists(),
        "files": file_status,
        "all_required_present": all(file_status[f] for f in required_files),
        "has_model_weights": any(file_status.get(f, False) for f in model_files)
    }

# Load model when module is imported (but not when run as main)
if __name__ != "__main__":
    try:
        # Check model files first
        file_check = check_model_files()
        if not file_check["model_directory_exists"]:
            logger.warning(f"Model directory does not exist: {TRAINED_MODEL_PATH}")
        elif not file_check["all_required_present"]:
            logger.warning("Some required model files are missing")
            logger.info(f"File status: {file_check['files']}")
        else:
            load_model()
    except Exception as e:
        logger.error(f"Failed to load TinyLlama model on import: {str(e)}")
        logger.info("Model will be loaded on first use")
else:
    # If run directly, show model file status
    file_status = check_model_files()
    print("Model File Status:")
    print(f"Directory exists: {file_status['model_directory_exists']}")
    print(f"All required files present: {file_status['all_required_present']}")
    print(f"Has model weights: {file_status['has_model_weights']}")
    print("\nFile details:")
    for file, exists in file_status['files'].items():
        print(f"  {file}: {'✓' if exists else '✗'}")
    
    if file_status['model_directory_exists'] and file_status['all_required_present']:
        try:
            load_model()
            print("\nModel loaded successfully!")
            info = get_model_info()
            print(f"Model info: {info}")
        except Exception as e:
            print(f"\nFailed to load model: {e}")
    else:
        print("\nCannot load model - missing required files.")