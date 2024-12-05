"""
Contradiction Detector Training Script
This script fine-tunes the Mixtral-8x7B model for contradiction detection using QLoRA.
"""
#Installing and Loading the required libraries 
# !pip install -q -U bitsandbytes transformers peft accelerate datasets scipy

import torch
import transformers
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import load_dataset

# Suppress warnings
warnings.filterwarnings("ignore")

def print_trainable_parameters(model):
    """Calculate and print the number of trainable parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")

def generate_prompt(user_query, sep="\n\n### "):
    """Generate a prompt for contradiction detection."""
    sys_msg = "Check if there is any contradiction in the deposition and highlight the type of contradiction."
    prompt = f"[INST]{sys_msg}\n{user_query['Contradiction']}[/INST]{user_query['Type']}"
    return prompt

def setup_tokenizer_and_model(model_name):
    """Initialize and configure the tokenizer and model."""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = "!"  # Set padding token
    
    # Configure quantization settings
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load and prepare the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    return tokenizer, model

def prepare_lora_config(model):
    """Configure and apply LoRA to the model."""
    # LoRA hyperparameters
    LORA_R = 8
    LORA_ALPHA = 2 * LORA_R
    LORA_DROPOUT = 0.1
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["w1"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    return get_peft_model(model, lora_config)

def prepare_dataset(tokenizer, data_path, max_length):
    """Load and prepare the dataset for training."""
    # Load dataset
    data = load_dataset('csv', data_files=data_path)
    train_data = data['train']
    
    # Tokenization function
    def tokenize(prompt):
        return tokenizer(
            prompt + tokenizer.eos_token,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    # Process dataset
    dataset = train_data.shuffle().map(
        lambda x: tokenize(generate_prompt(x)),
        remove_columns=["Contradiction", "Type"]
    )
    
    return dataset

def main():
    # Configuration
    MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    DATA_PATH = "/kaggle/input/d/alexwanjai/contradictions/Contradiction.csv"
    CUTOFF_LEN = 256
    OUTPUT_DIR = "mixtral-contradiction_detector"
    
    # Setup model and tokenizer
    tokenizer, base_model = setup_tokenizer_and_model(MODEL_NAME)
    model = prepare_lora_config(base_model)
    
    # Print trainable parameters
    print_trainable_parameters(model)
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer, DATA_PATH, CUTOFF_LEN)
    
    # Configure training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=6,
        learning_rate=1e-4,
        logging_steps=2,
        optim="adamw_torch",
        save_strategy="epoch",
        weight_decay=0.01,
        output_dir=OUTPUT_DIR
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False
        ),
    )
    
    # Disable caching for training
    model.config.use_cache = False
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()