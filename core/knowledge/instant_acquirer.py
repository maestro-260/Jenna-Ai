import requests

class InstantKnowledgeAcquirer:
    """Acquires and summarizes knowledge from new sources instantly (web/API)."""
    def __init__(self):
        pass

    def acquire_from_url(self, url: str) -> str:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.text[:10000]  # Limit for demo
        except Exception:
            return ""
        return ""
