from utils.config_loader import get_config, cached_config
from openvoice import ToneColorConverter
import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PersonalityEngine:

    def __init__(self):
        self.config = get_config("personality.yaml")
        self.reference_voice = "assets/voices/reference_audio.wav"
        if not os.path.exists(self.reference_voice):
            print(
                f"Error: Reference voice file {self.reference_voice} not found"
            )
        self.converter = ToneColorConverter(
            'checkpoints/converter', device="cuda"
        )
        # User personality profiles storage
        self.user_profiles = {}
        self.user_profiles_path = "memory/user_profiles.json"
        self._load_user_profiles()
        
    def _load_user_profiles(self):
        """Load stored user profiles or create empty profiles storage"""
        try:
            if os.path.exists(self.user_profiles_path):
                with open(self.user_profiles_path, 'r') as f:
                    self.user_profiles = json.load(f)
            else:
                self.user_profiles = {}
                # Ensure the directory exists
                os.makedirs(os.path.dirname(self.user_profiles_path), exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to load user profiles: {e}")
            self.user_profiles = {}
    
    def save_user_profiles(self):
        """Save user profiles to disk"""
        try:
            with open(self.user_profiles_path, 'w') as f:
                json.dump(self.user_profiles, f)
        except Exception as e:
            logger.error(f"Failed to save user profiles: {e}")

    async def generate_response(self, text: str, emotion: str, user_id: str = "default", memory_facts=None, reasoning=None, tool_result=None, kg_result=None) -> str:
        # Compose a prompt that includes memory facts, reasoning, tool and KG results
        prompt = f"User: {text}\n"
        if memory_facts:
            prompt += f"Relevant facts: {memory_facts}\n"
        if reasoning:
            prompt += f"Reasoning steps: {reasoning}\n"
        if tool_result:
            prompt += f"Tool result: {tool_result}\n"
        if kg_result:
            prompt += f"Knowledge graph: {kg_result}\n"
        prompt += f"Respond as a {emotion} assistant, adapting tone and style for user {user_id}."
        # Call LLM or TTS pipeline as needed (placeholder)
        # Here, you would use your LLM or TTS system to generate the actual response
        return prompt
        """Generate response tailored to the user's preferences and context"""
        style = self._select_style(emotion, user_id)
        return self.converter.convert(
            text=text, tone_color=self.reference_voice, style=style
        )

    def _select_style(self, emotion: str, user_id: str = "default") -> str:
        """Select appropriate style based on user preferences and context"""
        # Get user profile or create default
        user_profile = self.user_profiles.get(user_id, {
            "preferred_style": self.config["base_profile"],
            "emotional_preferences": {},
            "formality_level": 0.5,  # 0 = casual, 1 = formal
            "interactions": 0
        })
        
        # Use user-preferred style for this emotion if available
        if emotion in user_profile.get("emotional_preferences", {}):
            return user_profile["emotional_preferences"][emotion]
        
        # Otherwise use system default for this emotion
        return self.config["emotional_responses"].get(
            emotion, user_profile.get("preferred_style", self.config["base_profile"])
        )
    
    def update_user_preferences(self, user_id: str, interaction_data: Dict[str, Any]):
        """Update user preferences based on interaction data"""
        if not user_id or not interaction_data:
            return
            
        # Get or create user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "preferred_style": self.config["base_profile"],
                "emotional_preferences": {},
                "formality_level": 0.5,
                "interactions": 0,
                "created_at": datetime.now().isoformat()
            }
        
        user_profile = self.user_profiles[user_id]
        
        # Update interaction count
        user_profile["interactions"] = user_profile.get("interactions", 0) + 1
        user_profile["last_interaction"] = datetime.now().isoformat()
        
        # Update emotional preferences if feedback was positive
        if interaction_data.get("feedback", "").lower() in ["positive", "good", "like"]:
            emotion = interaction_data.get("emotion", "neutral")
            style = interaction_data.get("style", self.config["base_profile"])
            user_profile.setdefault("emotional_preferences", {})[emotion] = style
        
        # Adapt formality level based on user's language
        text = interaction_data.get("text", "").lower()
        formality_markers = {
            "formal": ["please", "would you", "could you", "thank you", "appreciate"],
            "casual": ["hey", "cool", "awesome", "thanks", "btw", "lol"]
        }
        
        # Count markers of each type
        formal_count = sum(1 for marker in formality_markers["formal"] if marker in text)
        casual_count = sum(1 for marker in formality_markers["casual"] if marker in text)
        
        # Gradually shift formality level (small adjustments over time)
        if formal_count > casual_count:
            user_profile["formality_level"] = min(1.0, user_profile.get("formality_level", 0.5) + 0.05)
        elif casual_count > formal_count:
            user_profile["formality_level"] = max(0.0, user_profile.get("formality_level", 0.5) - 0.05)
        
        # Save updated profiles
        self.save_user_profiles()
    
    def get_communication_style(self, user_id: str) -> Dict[str, Any]:
        """Get the current communication style for a user"""
        user_profile = self.user_profiles.get(user_id, {})
        formality = user_profile.get("formality_level", 0.5)
        
        # Determine concrete style parameters based on formality
        if formality > 0.7:
            style = "formal"
            sentence_length = "longer"
            vocabulary = "advanced"
        elif formality < 0.3:
            style = "casual"
            sentence_length = "shorter"
            vocabulary = "simple"
        else:
            style = "balanced"
            sentence_length = "medium"
            vocabulary = "standard"
            
        return {
            "style": style,
            "sentence_length": sentence_length,
            "vocabulary": vocabulary,
            "formality": formality,
            "preferred_emotion": self._get_preferred_emotion(user_id)
        }
    
    def _get_preferred_emotion(self, user_id: str) -> str:
        """Determine which emotional style the user responds to best"""
        user_profile = self.user_profiles.get(user_id, {})
        emotional_prefs = user_profile.get("emotional_preferences", {})
        
        # If no preferences yet, return neutral
        if not emotional_prefs:
            return "neutral"
            
        # Count occurrences of each emotion
        emotion_counts = {}
        for emotion in emotional_prefs:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
        # Return most frequently preferred emotion
        if emotion_counts:
            return max(emotion_counts, key=emotion_counts.get)
        return "neutral"