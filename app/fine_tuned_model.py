"""
Fine-tuned model integration for the RAG system
Allows switching between Gemini and fine-tuned models
"""

import os
from typing import Optional, Dict, Any
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from app.settings import settings

# Global model cache
_fine_tuned_model = None
_fine_tuned_tokenizer = None

def load_fine_tuned_model(model_path: str = "ft/out") -> tuple:
    """Load the fine-tuned model and tokenizer"""
    global _fine_tuned_model, _fine_tuned_tokenizer
    
    if _fine_tuned_model is None:
        print(f"Loading fine-tuned model from {model_path}...")
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fine-tuned model not found at {model_path}")
        
        # Load tokenizer and model
        _fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _fine_tuned_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        print("✅ Fine-tuned model loaded successfully!")
    
    return _fine_tuned_model, _fine_tuned_tokenizer

def create_fine_tuned_pipeline(model_path: str = "ft/out"):
    """Create a HuggingFace pipeline for the fine-tuned model"""
    model, tokenizer = load_fine_tuned_model(model_path)
    
    # Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0.1,
        do_sample=True,
        return_full_text=False
    )
    
    # Wrap in LangChain HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def get_model_info(model_path: str = "ft/out") -> Dict[str, Any]:
    """Get information about the fine-tuned model"""
    if not os.path.exists(model_path):
        return {"available": False, "error": "Model not found"}
    
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return {
            "available": True,
            "model_type": config.get("model_type"),
            "architecture": config.get("architectures", []),
            "vocab_size": config.get("vocab_size"),
            "hidden_size": config.get("hidden_size"),
            "num_layers": config.get("num_hidden_layers"),
            "path": model_path
        }
    
    return {"available": True, "path": model_path}

def format_instruction_prompt(instruction: str, input_text: str = "") -> str:
    """Format input for instruction-following fine-tuned model"""
    if input_text:
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""### Instruction:
{instruction}

### Response:
"""

class FineTunedModelManager:
    """Manager class for fine-tuned model operations"""
    
    def __init__(self, model_path: str = "ft/out"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
    
    def is_available(self) -> bool:
        """Check if fine-tuned model is available"""
        return os.path.exists(self.model_path)
    
    def load(self):
        """Load the fine-tuned model"""
        if not self.is_available():
            raise FileNotFoundError(f"Fine-tuned model not found at {self.model_path}")
        
        self.model, self.tokenizer = load_fine_tuned_model(self.model_path)
        self.pipeline = create_fine_tuned_pipeline(self.model_path)
    
    def generate(self, instruction: str, input_text: str = "", max_length: int = 1024) -> str:
        """Generate response using the fine-tuned model"""
        if self.pipeline is None:
            self.load()
        
        # Format the prompt
        prompt = format_instruction_prompt(instruction, input_text)
        
        # Generate response
        response = self.pipeline(prompt)
        
        # Extract the generated text
        if isinstance(response, str):
            return response
        elif isinstance(response, list) and len(response) > 0:
            return response[0].get('generated_text', '')
        else:
            return str(response)
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return get_model_info(self.model_path)

# Global fine-tuned model manager
fine_tuned_manager = FineTunedModelManager()