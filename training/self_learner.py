import logging
import os
import time
import asyncio
import subprocess
from typing import Optional, List, Tuple
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer

# Avoid circular import by importing the class, not the module


class SelfLearningEngine:
    """
    A self-learning engine for fine-tuning large language models
    with user-specific interactions and automatic model switching.
    """

    def __init__(self,
                 base_model: str = "mistralai/Mistral-7B-v0.1",
                 model_save_dir: str = "./models"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.base_model = base_model
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        # Initialize tokenizer lazily to avoid loading at startup
        self._tokenizer = None
        # Store recent interactions for learning
        self.recent_interactions = []
        # Lock for async operations
        self._finetune_lock = asyncio.Lock()

    @property
    def tokenizer(self):
        """Lazy load tokenizer when needed"""
        if self._tokenizer is None:
            self.logger.info(f"Loading tokenizer for {self.base_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def log_interaction(self, interaction: dict) -> None:
        """Store interactions for future learning"""
        if not interaction:
            return
        self.recent_interactions.append({
            'input': interaction.get('original_input', ''),
            'output': interaction.get('text', ''),
            'intent': interaction.get('intent', 'general'),
            'emotion': interaction.get('emotion', 'neutral'),
            'reward': interaction.get('reward', 0.0)
        })
        # Keep only recent 1000 interactions in memory
        if len(self.recent_interactions) > 1000:
            self.recent_interactions = self.recent_interactions[-1000:]

    def update_with_feedback(self, interaction: dict, reward: float) -> None:
        """Add explicit/implicit feedback to recent interactions for RLHF."""
        if not interaction:
            return
        entry = {
            'input': interaction.get('original_input', ''),
            'output': interaction.get('text', ''),
            'intent': interaction.get('intent', 'general'),
            'emotion': interaction.get('emotion', 'neutral'),
            'reward': reward
        }
        self.recent_interactions.append(entry)
        if len(self.recent_interactions) > 1000:
            self.recent_interactions = self.recent_interactions[-1000:]

    def trigger_finetune(self):
        """Trigger fine-tuning using recent feedback-weighted interactions."""
        # Gather only interactions with reward signals
        train_data = [
            (i['input'], i['output'], i['intent'], i['emotion'])
            for i in self.recent_interactions if abs(i.get('reward', 0.0)) > 0.0
        ]
        if len(train_data) >= 10:
            # Run fine-tuning asynchronously
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.finetune_on_data(train_data, user_id="default_user"))
            else:
                loop.run_until_complete(self.finetune_on_data(train_data, user_id="default_user"))

    async def create_dataset(
            self, training_data: List[Tuple]
            ) -> Optional[Dataset]:
        """Create a dataset from training data tuples"""
        if not training_data:
            self.logger.warning("No training data provided")
            return None
        try:
            return Dataset.from_dict({
                "text": [
                    f"<user>\n{interaction[0]}\n<assistant>\n{interaction[1]}"
                    for interaction in training_data
                ],
                "metadata": [
                    {"intent": interaction[2], "emotion": interaction[3]}
                    for interaction in training_data
                ]
            })
        except Exception as e:
            self.logger.error(f"Dataset creation failed: {e}")
            return None

    async def finetune_on_data(
            self,
            training_data: List[Tuple],
            user_id: str) -> Optional[str]:
        """
        Fine-tunes the model on user interactions, returns the new model name.
        """
        async with self._finetune_lock:
            try:
                # Generate unique model name with timestamp and user ID
                new_model_name = f"jenna-ft-{user_id[:8]}-{int(time.time())}"
                output_dir = os.path.join(self.model_save_dir, new_model_name)
                self.logger.info(f"Starting fine-tuning for {new_model_name}")
                # Create dataset from training data
                dataset = await self.create_dataset(training_data)
                if not dataset or len(dataset) < 10:
                    self.logger.warning(
                        "Insufficient training data, min 10 examples required"
                        )
                    return None
                # Run compute-intensive operations in thread pool
                result = await asyncio.to_thread(
                    self._run_finetune,
                    dataset=dataset,
                    output_dir=output_dir
                )
                if not result:
                    return None
                # Convert to GGUF format for Ollama compatibility
                gguf_success = await asyncio.to_thread(
                    self.convert_to_gguf,
                    model_path=os.path.join(output_dir, "final_model"),
                    new_model_name=new_model_name
                )
                if gguf_success:
                    self.logger.info(f"Fine-tuning complete: {new_model_name}")
                    return new_model_name
                return None
            except Exception as e:
                self.logger.error(f"Fine-tuning failed: {e}")
                return None

    def _run_finetune(self, dataset: Dataset, output_dir: str) -> bool:
        """Run the actual fine-tuning process (CPU/GPU intensive)"""
        try:
            # Load model with appropriate precision based on hardware
            torch_dtype = torch.float16 if torch.cuda.is_available() else \
                torch.float32
            device_map = "auto" if torch.cuda.is_available() else None
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True
            )
            # Configure LoRA for efficient fine-tuning
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            # Configure training arguments based on available hardware
            batch_size = 2 if torch.cuda.is_available() else 1
            grad_accum = 4 if torch.cuda.is_available() else 8
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=5e-5,
                weight_decay=0.01,
                max_grad_norm=0.3,
                num_train_epochs=3,
                warmup_steps=100,
                logging_steps=10,
                fp16=torch.cuda.is_available(),
                push_to_hub=False,
                remove_unused_columns=False
            )
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                tokenizer=self.tokenizer,
                peft_config=peft_config,
                dataset_text_field="text",
                max_seq_length=512,
                args=training_args
            )
            trainer.train()
            final_model_path = os.path.join(output_dir, "final_model")
            trainer.save_model(final_model_path)
            return True
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return False

    def convert_to_gguf(self, model_path: str, new_model_name: str) -> bool:
        """
        Converts the fine-tuned model to GGUF format and registers with Ollama.
        """
        gguf_output = os.path.join(
            self.model_save_dir, f"{new_model_name}.gguf"
        )
        # Find the converter script
        script_paths = [
            "llama.cpp/convert-hf-to-gguf.py",
            "tools/convert-hf-to-gguf.py",
            os.path.expanduser("~/.local/bin/convert-hf-to-gguf.py")
        ]
        convert_script = None
        for path in script_paths:
            if os.path.exists(path):
                convert_script = path
                break
        if not convert_script:
            self.logger.error("GGUF conversion script not found")
            return False
        try:
            self.logger.info(f"Converting {new_model_name} to GGUF format...")
            convert_cmd = [
                "python", convert_script,
                model_path,
                "--outfile", gguf_output,
                "--outtype", "f16"
            ]
            result = subprocess.run(
                convert_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            if not os.path.exists(gguf_output):
                self.logger.error(
                    f"GGUF conversion failed: {result.stderr}"
                    )
                return False
            self.logger.info(
                f"Registering model {new_model_name} with Ollama..."
                )
            # Create Modelfile for Ollama
            modelfile_path = os.path.join(
                self.model_save_dir, f"{new_model_name}.modelfile"
                )
            with open(modelfile_path, "w") as f:
                f.write(f"FROM {gguf_output}\n")
                f.write("PARAMETER temperature 0.7\n")
                f.write("PARAMETER top_p 0.9\n")
                f.write("PARAMETER stop <user>\n")
                f.write("PARAMETER stop <assistant>\n")
            # Register with Ollama
            ollama_cmd = f"ollama create {new_model_name} -f {modelfile_path}"
            subprocess.run(ollama_cmd, shell=True, check=True)
            self.logger.info(f"Model {new_model_name} successfully registered")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ollama registration failed: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"GGUF conversion error: {str(e)}")
            return False
