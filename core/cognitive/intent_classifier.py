import asyncio
import torch  # Ensured PyTorch is installed: pip install torch
from utils.config_loader import get_config, cached_config
import logging
from datasets import Dataset
from transformers import (TrainingArguments, Trainer,
                          DistilBertTokenizer,
                          DistilBertForSequenceClassification)
from rasa.nlu.model import Interpreter

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentClassifier:
    def __init__(self):
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=7
            ).to(self.device)
            self.labels = self._load_labels()
            try:
                self.rasa_interpreter = Interpreter.load("./rasa_model")
            except Exception as e:
                logger.error(f"Failed to load Rasa model: {e}")
                self.rasa_interpreter = None
                self.labels = ["query", "web_search", "ecommerce", "calculation",
                             "navigation", "form_fill", "system_command"]
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def _load_labels(self):
        return get_config("intents.yaml").get("intents", ["unknown"])

    async def classify(self, text: str) -> dict:
        # Use Rasa for intent and entity extraction
        rasa_result = await asyncio.to_thread(
            self.rasa_interpreter.parse, text
            )
        intent = rasa_result["intent"]["name"]
        entities = {
            entity["entity"]: entity["value"]
            for entity in rasa_result["entities"]
        }
        
        # Fallback to DistilBERT if Rasa fails
        if not intent or intent not in self.labels:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)
            logits = await asyncio.to_thread(
                lambda: self.model(**inputs).logits
                )
            intent = self.labels[torch.argmax(logits).item()]
        
        return {"intent": intent, "entities": entities}
    
    async def fine_tune(self, training_data: list):
        dataset = Dataset.from_dict({
            "text": [item[0] for item in training_data],
            "label": [
                self.labels.index(item[1]) if item[1] in self.labels else 0
                for item in training_data
            ]
        })
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length",
                                  truncation=True, max_length=512)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        train_dataset = tokenized_dataset.shuffle(seed=42)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./intent_classifier_finetuned",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=500,
            save_total_limit=2,
            logging_dir='./logs',
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        await asyncio.to_thread(trainer.train)
        self.model.save_pretrained("./intent_classifier_finetuned")
        self.tokenizer.save_pretrained("./intent_classifier_finetuned")
    
        logger.info("Intent classifier fine-tuned and saved.")