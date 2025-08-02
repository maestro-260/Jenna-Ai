import re
import logging
from typing import Dict
from transformers import pipeline
from utils.config_loader import get_config, cached_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EthicsGuardian:
    def __init__(self, bias_threshold: float = 0.75):
        self.bias_threshold = bias_threshold
        self.constitution = get_config("constraints.yaml")
        try:
            self.bias_detector = pipeline(
                "text-classification",
                model="facebook/roberta-hate-speech-detection",
                truncation=True,
            )
            logger.info(
                ("Bias detection model loaded. Remember to update the model "
                 "periodically.")
            )
        except Exception as e:
            logger.error(f"Error loading bias detection model: {e}")
            self.bias_detector = None
            
        # Retain regex-based privacy detection
        self.patterns = {
            'privacy': re.compile(
                r"\b(ssn|phone|address|password|credit card)\b", re.IGNORECASE
            ),
            'dangerous_commands': re.compile(
                r"\b(sudo|rm -rf|chmod|format|del|erase)\b", re.IGNORECASE
            ),
        }
        
    def audit_response(self, response: str, context: Dict) -> Dict:
        """
        Audit a response for ethical concerns based on content and context.
        
        Args:
            response: The text response to audit
            context: Contextual information about the interaction
            
        Returns:
            A report of potential ethical issues
        """
        report = {
            'bias_detected': self._check_bias(response),
            'privacy_leaks': self._check_privacy(response),
            'safety_risk': self._check_safety(context),
            'pass': True,  # Default to passing the audit
        }
        
        # Mark as failing if any issues detected
        if any([
            report['bias_detected'], 
            report['privacy_leaks'], 
            report['safety_risk']
        ]):
            report['pass'] = False
            logger.warning(f"Ethics audit failed: {report}")
        
        return report
        
    def _check_bias(self, text: str) -> bool:
        """
        Check if the text contains potentially biased or hateful content.
        
        Args:
            text: The text to check
            
        Returns:
            Boolean indicating if bias was detected
        """
        # Skip if model failed to load
        if self.bias_detector is None:
            logger.warning(
                "Bias detector model not loaded. Skipping bias check."
                )
            return False
            
        try:
            # Use transformer-based pipeline to evaluate potential hate speech
            results = self.bias_detector(text)
            
            for result in results:
                label = result.get("label", "").lower()
                score = result.get("score", 0)
                
                if (
                    label in ["hate", "hate_speech", "offensive"]
                    and score > self.bias_threshold
                ):
                    logger.info(f"Bias detected: {label} with score {score}")
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error in bias detection: {e}")
            return False
            
    def _check_privacy(self, text: str) -> bool:
        """
        Check for privacy leaks using regex patterns.
        
        Args:
            text: The text to check
            
        Returns:
            Boolean indicating if privacy concerns were detected
        """
        try:
            # Check for privacy leaks using regex patterns
            if self.patterns['privacy'].search(text):
                logger.info("Privacy leak detected in text")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in privacy check: {e}")
            return False
            
    def _check_safety(self, context: Dict) -> bool:
        """
        Evaluate safety risk based on context.
        
        Args:
            context: Context information about the interaction
            
        Returns:
            Boolean indicating if safety risk was detected
        """
        # Check for emotional indicators of risk
        emotional_risk = context.get('emotion') in ['angry', 'violent']
        
        # Check for physical action requests
        physical_action = context.get('physical_action_requested', False)
        
        # Check if action intensity exceeds safety threshold
        if 'action' in context and 'intensity' in context['action']:
            intensity_risk = (
                context['action']['intensity']
                > self.constraints['safety'].get('max_intensity', 0.7)
            )
        else:
            intensity_risk = False
            
        # Return true if any risk factors are present
        return emotional_risk or (physical_action or intensity_risk)