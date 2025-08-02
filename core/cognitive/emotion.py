from transformers import pipeline
import torch
import ollama
from typing import Dict, Optional
import logging
import asyncio


class EmotionAnalyzer:
    """Analyzes emotion in text and generates empathetic responses."""
    
    def __init__(self, active_model: str):
        """Initialize the emotion analyzer.
        
        Args:
            active_model: Name of the active model used for response generation
        """
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if torch.cuda.is_available() else -1
        )
        self.active_model = active_model
        self.logger = logging.getLogger(__name__)

    async def analyze(
            self, text: str, prosody: Optional[Dict[str, float]] = None
            ) -> str:
        """Analyze the emotion in text and prosody data.
        
        Args:
            text: The text to analyze
            prosody: Optional prosody features including pitch_range
            
        Returns:
            The dominant emotion as a string
        """
        results = await asyncio.to_thread(self.classifier, text)
        dominant_emotion = max(results, key=lambda x: x["score"])["label"]
        
        # Override with excited if neutral but high pitch
        if (prosody and isinstance(prosody, dict) and 
                "pitch_range" in prosody and prosody["pitch_range"] > 0.5 and 
                dominant_emotion == "neutral"):
            dominant_emotion = "excited"
            
        return dominant_emotion

    async def generate_empathetic_reply(
            self,
            text: str,
            prosody: Optional[Dict[str, float]] = None,
            emotion_override: Optional[str] = None
    ) -> str:
        """Generate an empathetic reply based on detected emotion.
        
        Args:
            text: The user input text
            prosody: Optional prosody features
            emotion_override: Optional emotion to override detection
            
        Returns:
            An empathetic response string
        """
        dominant_emotion = (
            emotion_override 
            if emotion_override 
            else await self.analyze(text, prosody)
        )
        
        prompt = (
            f"The user said: \"{text}\"\n"
            f"Detected emotion: {dominant_emotion}.\n"
            "Respond with empathy and encouragement, "
            "acknowledging their emotional state."
        )
        
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.active_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an empathetic assistant."
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            return response["message"]["content"]
        except Exception as e:
            self.logger.error(f"Error generating empathetic reply: {e}")
            return (
                f"I'm here for you. "
                f"How can I assist with {dominant_emotion}?"
            )