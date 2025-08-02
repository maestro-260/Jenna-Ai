import re
from transformers import pipeline
import torch


class SecurityMonitor:
    def __init__(self):
        self.detector = pipeline(
            "text-classification",
            model="microsoft/deberta-v3-base-threat-detection",
            device=0 if torch.cuda.is_available() else -1
        )
        self.patterns = {
            'injection': r'(sudo|rm -rf|chmod)',
            'privacy': r'\b(ssn|password|credit_card)\b',
            'malicious': r'\b(hack|exploit|phish|malware)\b'
        }

    def analyze(self, text: str) -> dict:
        threat_score = self._neural_analysis(text)
        pattern_matches = self._pattern_scan(text)
        recommendation = self._generate_recommendation(text)
        is_threat = threat_score > 0.5 or bool(pattern_matches)
        return {
            'threat_score': threat_score,
            'pattern_matches': pattern_matches,
            'recommendation': recommendation,
            'is_threat': is_threat
        }

    def _neural_analysis(self, text: str) -> float:
        try:
            results = self.detector(text)
            return results[0]['score'] if results else 0.0
        except Exception:
            return 0.0

    def _pattern_scan(self, text: str) -> list:
        matches = [
            pattern for pattern, regex in self.patterns.items()
            if re.search(regex, text, re.IGNORECASE)
        ]
        return matches

    def _generate_recommendation(self, text: str) -> str:
        if self._pattern_scan(text):
            return "Avoid using system commands or sensitive keywords."
        return "Input appears safe."

    async def check_input(self, text: str) -> bool:
        """Check if input is safe to process"""
        if not text or not isinstance(text, str):
            return False
            
        analysis = self.analyze(text)
        return not analysis['is_threat']