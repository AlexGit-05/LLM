"""
Deposition Summarization Script
This script uses Mixtral model to generate concise summaries of legal depositions,
focusing on key information like case details, topics discussed, and crucial statements.
"""
# !pip install -U transformers
# !pip install -U accelerate
# !pip install -U bitsandbytes
# !pip install -U langchain
# !pip install -U langchain_community
# !pip install modelbit

import torch
import warnings
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline
)
from langchain import HuggingFacePipeline

# Suppress warnings
warnings.filterwarnings('ignore')

class DepositionSummarizer:
    def __init__(self, model_path: str):
        """
        Initialize the summarizer with the specified model path.
        
        Args:
            model_path (str): Path to the Mixtral model
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.llm = None
        
        # Initialize components
        self._setup_model()
        
    def _setup_model(self):
        """Set up the model, tokenizer, and pipeline."""
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config
        )
        
        # Configure generation settings
        generation_config = GenerationConfig.from_pretrained(self.model_path)
        generation_config.max_new_tokens = 1024
        generation_config.temperature = 0.7
        generation_config.top_p = 0.95
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.15
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=True,
            generation_config=generation_config,
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
    
    def summarize(self, deposition_text: str) -> str:
        """
        Generate a summary of the provided deposition text.
        
        Args:
            deposition_text (str): The deposition text to summarize
            
        Returns:
            str: Generated summary
        """
        template = """[INST] Summarize the following deposition in approximately 50 words. Include the following:
- Deponent's name
- Case details (name, number, court, date, location)
- General topic of the case
- Key topics discussed
- Pivotal statements
- Significant objections and rulings
- Crucial discrepancies or admissions
- Exhibits referenced
[/INST] """
        
        # Generate summary
        formatted_text = template.format(text=deposition_text)
        result = self.pipe(formatted_text)
        
        # Extract summary from result
        output = result[0]['generated_text']
        start_marker = "[INST]"
        end_marker = "[/INST]"
        start_index = output.find(start_marker)
        end_index = output.rfind(end_marker)
        summary = output[end_index + 1:].strip()
        
        return summary

def main():
    """Main execution function."""
    # Example usage
    MODEL_PATH = "path/to/mixtral/model"  # Update with actual model path
    
    # Initialize summarizer
    summarizer = DepositionSummarizer(MODEL_PATH)
    
    # Read deposition text (example)
    with open("deposition.txt", "r") as f:
        deposition_text = f.read()
    
    # Generate summary
    summary = summarizer.summarize(deposition_text)
    print("Deposition Summary:")
    print(summary)

if __name__ == "__main__":
    main()