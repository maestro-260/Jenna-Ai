# core/utils/data_cleaner.py
import re
from typing import Dict


class DataSanitizer:
    def __init__(self):
        self.patterns = {
            'personal_info': (
                r'\b\d{3}-\d{2}-\d{4}\b|'  # SSN
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|'  
                # Email
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|'  # Phone
                r'\b\d{16}\b|'  # Credit card
                r'\b\d+ [A-Za-z0-9\s,]+ (?:Road|Street|Ave|Boulevard|'
                r'Blvd|Lane|Ln)\b'  # Address
            ),
            'profanity': r'\b(asshole|bastard|shit)\b',
            'system_commands': r'\b(rm\s+-rf|sudo|chmod|format|del)\b'
        }

    def sanitize_text(self, text: str) -> str:
        """Remove sensitive information from text"""
        cleaned = text
        for _, pattern in self.patterns.items():
            cleaned = re.sub(
                pattern, '[REDACTED]', cleaned, flags=re.IGNORECASE
            )
        return cleaned

    def filter_interactions(self, interaction: Dict) -> Dict:
        """Sanitize and validate a single interaction"""
        return {
            'input': self.sanitize_text(interaction['input']),
            'response': self.sanitize_text(interaction['response']),
            'intent': interaction['intent']
        }