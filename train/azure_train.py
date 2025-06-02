#!/usr/bin/env python3
"""
TinyLlama Fine-tuning Script for Tech News Chat Engine - Azure ML Version (CPU-only)
Trains a TinyLlama model on conversational data from Azure Blob Storage
Modified to use TinyLlama-1.1B-Chat-v1.0-GPTQ without quantization and CPU-only training
"""

import os
import json
import torch
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import shutil

# Azure ML imports
import azureml.core
from azureml.core import Run, Dataset, Datastore, Workspace
from azure.storage.blob import BlobServiceClient

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    
    # Model settings - Using TheBloke TinyLlama from Hugging Face
    model_name: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"
    model_container_name: str = "llama-model"
    model_blob_path: str = "tinyllama"  # Path in blob container for the model
    local_model_path: str = "./tinyllama_model"  # Local path to save/load model
    
    # Disable quantization and force CPU
    use_4bit: bool = False
    use_lora: bool = True
    device: str = "cpu"  # Force CPU usage
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Training settings - adjusted for CPU training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # Reduced for CPU
    per_device_eval_batch_size: int = 1   # Reduced for CPU
    gradient_accumulation_steps: int = 16  # Increased to maintain effective batch size
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Generation settings
    max_seq_length: int = 1024  # Reduced for CPU efficiency
    
    # Azure settings
    blob_container_name: str = "tech-news-data"
    blob_data_path: str = "training"
    
    # Local paths in Azure ML
    data_dir: str = "."
    output_dir: str = "./outputs"  # Azure ML outputs folder
    logging_steps: int = 20
    save_steps: int = 200
    eval_steps: int = 200
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01

class AzureTinyLlamaTrainer:
    """Main training class for TinyLlama fine-tuning on Azure ML (CPU-only)"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.run = None
        self.workspace = None
        self.connection_string = None
        
        # Force CPU usage
        torch.cuda.is_available = lambda: False
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Get Azure ML run context
        try:
            self.run = Run.get_context()
            self.workspace = self.run.experiment.workspace
            logger.info("Azure ML run context acquired successfully")
        except Exception as e:
            logger.warning(f"Could not get Azure ML context: {str(e)}")
            self.run = None
        
        # Get Azure Storage connection string
        self.connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable must be set")
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.local_model_path, exist_ok=True)

    def download_or_load_model(self):
        """Download TinyLlama model from Hugging Face and optionally from/to blob storage"""
        logger.info(f"Loading TinyLlama model: {self.config.model_name}")
        
        try:
            # First try to download from blob storage if it exists
            if self.check_model_in_blob():
                logger.info("Found model in Azure Blob Storage, downloading...")
                return self.download_model_from_blob()
            else:
                logger.info("Model not found in blob storage, downloading from Hugging Face...")
                return self.download_model_from_hf()
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def check_model_in_blob(self) -> bool:
        """Check if model exists in Azure Blob Storage"""
        try:
            blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            container_client = blob_service_client.get_container_client(self.config.model_container_name)
            
            # Check if model files exist
            blob_list = list(container_client.list_blobs(name_starts_with=self.config.model_blob_path))
            return len(blob_list) > 0
            
        except Exception:
            return False

    def download_model_from_hf(self) -> str:
        """Download TinyLlama model from Hugging Face and save to blob storage"""
        logger.info(f"Downloading {self.config.model_name} from Hugging Face...")
        
        model_path = Path(self.config.local_model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download tokenizer and model from Hugging Face
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Save tokenizer locally
            tokenizer.save_pretrained(str(model_path))
            logger.info("Tokenizer downloaded and saved")
            
            # Download model without quantization config
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map=None,  # Don't use device mapping
                trust_remote_code=True
            )
            
            # Save model locally
            model.save_pretrained(str(model_path))
            logger.info("Model downloaded and saved")
            
            # Upload to blob storage for future use
            self.upload_model_to_blob(str(model_path), is_base_model=True)
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download model from Hugging Face: {str(e)}")
            raise

    def download_model_from_blob(self) -> str:
        """Download the TinyLlama model from Azure Blob Storage"""
        logger.info(f"Downloading model from Azure Blob Storage container: {self.config.model_container_name}")
        
        try:
            blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            container_client = blob_service_client.get_container_client(self.config.model_container_name)
            
            # Create local model directory
            model_path = Path(self.config.local_model_path)
            model_path.mkdir(parents=True, exist_ok=True)
            
            # List all blobs in the model container with the specified path
            blob_list = container_client.list_blobs(name_starts_with=self.config.model_blob_path)
            
            downloaded_files = []
            for blob in blob_list:
                # Skip directories (blobs ending with /)
                if blob.name.endswith('/'):
                    continue
                
                blob_client = container_client.get_blob_client(blob.name)
                
                # Create local file path
                relative_path = blob.name[len(self.config.model_blob_path):].lstrip('/')
                if not relative_path:  # If blob_path matches exactly
                    relative_path = Path(blob.name).name
                
                local_file_path = model_path / relative_path
                
                # Create directories if needed
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download the blob
                logger.info(f"Downloading: {blob.name} -> {local_file_path}")
                with open(local_file_path, 'wb') as download_file:
                    blob_data = blob_client.download_blob()
                    download_file.write(blob_data.readall())
                
                downloaded_files.append(str(local_file_path))
            
            if not downloaded_files:
                raise ValueError(f"No model files found in container '{self.config.model_container_name}' with prefix '{self.config.model_blob_path}'")
            
            logger.info(f"Successfully downloaded {len(downloaded_files)} model files to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download model from blob storage: {str(e)}")
            raise

    def download_data_from_blob(self):
        """Download training data from Azure Blob Storage"""
        logger.info("Downloading training data from Azure Blob Storage...")
        
        try:
            # Get the default datastore (should be configured to point to your blob storage)
            if self.workspace:
                datastore = Datastore.get_default(self.workspace)
                logger.info(f"Using datastore: {datastore.name}")
                
                # Download data from blob storage
                target_path = Path(self.config.data_dir) / "training"
                target_path.mkdir(parents=True, exist_ok=True)
                
                # Create a dataset reference to the blob path
                try:
                    dataset = Dataset.File.from_files(
                        path=[(datastore, f"{self.config.blob_data_path}/*")]
                    )
                    
                    # Download the files
                    dataset.download(target_path=str(target_path), overwrite=True)
                    logger.info(f"Data downloaded to: {target_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to download from dataset: {str(e)}")
                    # Fallback: try direct blob access
                    self._download_blob_direct()
            else:
                logger.warning("No workspace available, trying direct blob access")
                self._download_blob_direct()
                
        except Exception as e:
            logger.error(f"Failed to download data from blob storage: {str(e)}")
            raise

    def _download_blob_direct(self):
        """Direct blob storage download as fallback"""
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        container_client = blob_service_client.get_container_client(self.config.blob_container_name)
        
        target_path = Path(self.config.data_dir) / "training"
        target_path.mkdir(parents=True, exist_ok=True)
        
        # List and download blobs
        blob_list = container_client.list_blobs(name_starts_with=self.config.blob_data_path)
        
        for blob in blob_list:
            if blob.name.endswith('.json'):
                blob_client = container_client.get_blob_client(blob.name)
                download_path = target_path / Path(blob.name).name
                
                with open(download_path, 'wb') as download_file:
                    blob_data = blob_client.download_blob()
                    download_file.write(blob_data.readall())
                
                logger.info(f"Downloaded: {blob.name} to {download_path}")

    def load_model_and_tokenizer(self):
        """Load the TinyLlama model and tokenizer (CPU-only, no quantization)"""
        logger.info("Loading TinyLlama model and tokenizer for CPU training...")
        
        # Download or load model
        model_path = self.download_or_load_model()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        logger.info("Successfully loaded tokenizer")
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model for CPU training (no quantization)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,  # No device mapping for CPU
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Ensure model is on CPU
        self.model = self.model.to("cpu")
        logger.info("Successfully loaded model for CPU training")
        
        # Add LoRA adapters if specified
        if self.config.use_lora:
            # Get the correct target modules for TinyLlama
            target_modules = self._get_target_modules()
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        logger.info("Model and tokenizer loaded successfully for CPU training")

    def _get_target_modules(self):
        """Get target modules for LoRA based on TinyLlama architecture"""
        # TinyLlama target modules (similar to Llama but adapted for TinyLlama)
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        
        # Check what modules are actually available in this model
        available_modules = []
        for name, module in self.model.named_modules():
            module_name = name.split('.')[-1]
            if module_name in target_modules and module_name not in available_modules:
                available_modules.append(module_name)
        
        if available_modules:
            logger.info(f"Using LoRA target modules: {available_modules}")
            return available_modules
        else:
            # Fallback - check for common linear layer names
            fallback_targets = ["q_proj", "v_proj", "o_proj"]
            logger.warning(f"No standard target modules found, using fallback: {fallback_targets}")
            return fallback_targets

    def load_training_data(self) -> DatasetDict:
        """Load and prepare training data from downloaded files"""
        logger.info("Loading training data...")
        
        # Look for training data in the downloaded directory
        data_dir = Path(self.config.data_dir)
        llama_dir = data_dir / "training"
        
        if not llama_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {llama_dir}")
        
        # Look for train/val split files first
        train_files = list(llama_dir.glob("train_data_*.json"))
        val_files = list(llama_dir.glob("val_data_*.json"))
        
        if train_files and val_files:
            # Use existing train/val split
            latest_train = max(train_files, key=os.path.getctime)
            latest_val = max(val_files, key=os.path.getctime)
            
            logger.info(f"Loading train data from: {latest_train}")
            logger.info(f"Loading val data from: {latest_val}")
            
            with open(latest_train, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            
            with open(latest_val, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
        else:
            # Load full training data and create split
            training_files = list(llama_dir.glob("llama_training_data_*.json"))
            
            if not training_files:
                raise FileNotFoundError(f"No training data files found in: {llama_dir}")
            
            latest_file = max(training_files, key=os.path.getctime)
            logger.info(f"Loading training data from: {latest_file}")
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            # Create train/val split (90/10)
            split_idx = int(len(all_data) * 0.9)
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]
        
        logger.info(f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples")
        
        # Log to Azure ML if available
        if self.run:
            self.run.log("train_examples", len(train_data))
            self.run.log("val_examples", len(val_data))
        
        # Convert to datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        
        return dataset_dict

    def format_conversation(self, example: Dict) -> str:
        """Format conversation for TinyLlama training"""
        if "messages" in example:
            # ChatML format for TinyLlama
            messages = example["messages"]
            formatted = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    formatted += f"<|system|>\n{msg['content']}</s>\n"
                elif msg["role"] == "user":
                    formatted += f"<|user|>\n{msg['content']}</s>\n"
                elif msg["role"] == "assistant":
                    formatted += f"<|assistant|>\n{msg['content']}</s>\n"
            
            return formatted
        else:
            # Alpaca format adapted for TinyLlama
            system = example.get("system", "You are a helpful AI assistant.")
            instruction = example.get("instruction", example.get("user", ""))
            output = example.get("output", example.get("assistant", ""))
            
            formatted = f"<|system|>\n{system}</s>\n<|user|>\n{instruction}</s>\n<|assistant|>\n{output}</s>\n"
            return formatted

    def preprocess_function(self, examples):
        """Preprocess examples for training"""
        texts = []
        
        for example in examples:
            formatted_text = self.format_conversation(example)
            texts.append(formatted_text)
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.config.max_seq_length,
            return_tensors=None
        )
        
        # Set labels for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized

    def prepare_datasets(self, dataset_dict: DatasetDict):
        """Prepare datasets for training"""
        logger.info("Preprocessing datasets...")
        
        # Process training set
        train_examples = []
        for example in dataset_dict["train"]:
            train_examples.append(example)
        
        val_examples = []
        for example in dataset_dict["validation"]:
            val_examples.append(example)
        
        # Tokenize
        train_tokenized = self.preprocess_function(train_examples)
        val_tokenized = self.preprocess_function(val_examples)
        
        # Create new datasets
        train_dataset = Dataset.from_dict(train_tokenized)
        val_dataset = Dataset.from_dict(val_tokenized)
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })

    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments for CPU training"""
        return TrainingArguments(
            output_dir=f"{self.config.output_dir}/checkpoints",
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            fp16=False,  # Disable mixed precision for CPU
            use_cpu=True,  # Force CPU usage
            no_cuda=True,  # Disable CUDA
            push_to_hub=False,
            hub_model_id=None,
            dataloader_num_workers=0,  # Reduce for CPU training
        )

    def upload_model_to_blob(self, model_path: str, is_base_model: bool = False):
        """Upload trained model to Azure Blob Storage"""
        blob_path = self.config.model_blob_path if is_base_model else f"trained_models/tinyllama_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Uploading model to Azure Blob Storage at path: {blob_path}")
        
        try:
            if self.workspace and not is_base_model:
                # Use Azure ML datastore for trained models
                datastore = Datastore.get_default(self.workspace)
                
                # Upload the entire model directory
                datastore.upload(
                    src_dir=model_path,
                    target_path=blob_path,
                    overwrite=True,
                    show_progress=True
                )
                logger.info("Model uploaded to datastore successfully")
                
            else:
                # Direct blob upload
                self._upload_model_direct(model_path, blob_path)
                
        except Exception as e:
            logger.error(f"Failed to upload model: {str(e)}")
            if not is_base_model:
                # Don't raise for trained models - model is still saved locally
                pass
            else:
                raise

    def _upload_model_direct(self, model_path: str, blob_path: str):
        """Direct blob upload"""
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        container_name = self.config.model_container_name
        
        for root, dirs, files in os.walk(model_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                blob_file_path = os.path.join(
                    blob_path,
                    os.path.relpath(local_file_path, model_path)
                ).replace("\\", "/")
                
                blob_client = blob_service_client.get_blob_client(
                    container=container_name,
                    blob=blob_file_path
                )
                
                with open(local_file_path, 'rb') as data:
                    blob_client.upload_blob(data, overwrite=True)
                
                logger.info(f"Uploaded: {blob_file_path}")

    def train(self):
        """Main training function"""
        logger.info("Starting TinyLlama fine-tuning (CPU-only)...")
        
        # Download data from blob storage
        self.download_data_from_blob()
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Load and prepare data
        raw_datasets = self.load_training_data()
        processed_datasets = self.prepare_datasets(raw_datasets)
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8
        )
        
        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_datasets["train"],
            eval_dataset=processed_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            ]
        )
        
        # Start training
        logger.info("Beginning training on CPU...")
        trainer.train()
        
        # Save final model locally (Azure ML outputs)
        final_model_dir = f"{self.config.output_dir}/final_model"
        logger.info(f"Saving final model to: {final_model_dir}")
        
        trainer.save_model(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)
        
        # Save training stats
        training_stats = {
            "config": self.config.__dict__,
            "final_train_loss": trainer.state.log_history[-1].get("train_loss", "N/A") if trainer.state.log_history else "N/A",
            "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", "N/A") if trainer.state.log_history else "N/A",
            "total_steps": trainer.state.global_step,
            "training_time": str(datetime.now()),
            "base_model": self.config.model_name,
            "device": "CPU",
            "quantization": "None"
        }
        
        with open(f"{final_model_dir}/training_stats.json", 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        # Log metrics to Azure ML
        if self.run:
            self.run.log("final_train_loss", training_stats["final_train_loss"])
            self.run.log("final_eval_loss", training_stats["final_eval_loss"])
            self.run.log("total_steps", training_stats["total_steps"])
            self.run.log("base_model", self.config.model_name)
            self.run.log("device", "CPU")
            self.run.log("quantization", "None")
        
        # Upload model to blob storage
        self.upload_model_to_blob(final_model_dir)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final model saved to: {final_model_dir}")
        logger.info(f"Base model: {self.config.model_name}")
        logger.info("Training performed on CPU without quantization")
        
        return trainer

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama model for tech news chat on Azure ML (CPU-only)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
                       help="Hugging Face model name")
    parser.add_argument("--model_container", type=str, default="llama-model",
                       help="Azure Blob Storage container name for the model")
    parser.add_argument("--model_blob_path", type=str, default="tinyllama",
                       help="Path to model files in blob container")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for efficient fine-tuning")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size per device (CPU)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                       help="Maximum sequence length")
    
    # Azure arguments
    parser.add_argument("--blob_container", type=str, default="tech-news-data",
                       help="Azure Blob Storage container name for training data")
    parser.add_argument("--blob_data_path", type=str, default="training",
                       help="Path to training data in blob storage")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_name=args.model_name,
        model_container_name=args.model_container,
        model_blob_path=args.model_blob_path,
        use_lora=args.use_lora,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        blob_container_name=args.blob_container,
        blob_data_path=args.blob_data_path
    )
    
    # Initialize trainer
    trainer = AzureTinyLlamaTrainer(config)
    
    try:
        # Start training
        logger.info("Starting TinyLlama fine-tuning on Azure ML (CPU-only)")
        logger.info(f"Configuration: {config}")
        
        trained_model = trainer.train()
        
        logger.info("Training completed successfully!")
        return trained_model
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        # Log error to Azure ML if available
        if trainer.run:
            trainer.run.log("training_error", str(e))
            trainer.run.fail(error_details=str(e))
        raise
    finally:
        # Clean up temporary files if needed
        temp_dirs = [config.local_model_path, f"{config.data_dir}/training"]
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {temp_dir}: {str(e)}")


if __name__ == "__main__":
    main()