import os
from typing import List, Dict

class UnstructuredLearner:
    """Learns from unstructured data files (txt, md, pdf, web pages, etc.)."""
    def __init__(self):
        pass

    def extract_text(self, file_path: str) -> str:
        # For simplicity, only txt and md supported here. Extend for PDF/HTML as needed.
        if file_path.endswith(".txt") or file_path.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        # Add PDF/HTML extraction here as needed
        return ""

    def learn_from_files(self, file_paths: List[str]) -> Dict[str, str]:
        knowledge = {}
        for path in file_paths:
            text = self.extract_text(path)
            if text:
                knowledge[path] = text
        return knowledge
