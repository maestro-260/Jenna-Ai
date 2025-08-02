import json
from datetime import datetime
from typing import Dict, Any

class AnalyticsEngine:
    """Performs in-depth analysis and reporting on user data and interactions."""
    def __init__(self, log_path="memory/interaction_logs.json"):
        self.log_path = log_path
        self._ensure_log_file()

    def _ensure_log_file(self):
        try:
            with open(self.log_path, "a+") as f:
                pass
        except Exception:
            with open(self.log_path, "w") as f:
                json.dump([], f)

    def log_interaction(self, interaction: Dict[str, Any]):
        try:
            with open(self.log_path, "r+") as f:
                data = json.load(f)
                data.append(interaction)
                f.seek(0)
                json.dump(data, f)
                f.truncate()
        except Exception:
            pass

    def log_feedback(self, user_id: str, reward: float, feedback_type: str, timestamp=None):
        if not timestamp:
            timestamp = datetime.now().isoformat()
        feedback_entry = {
            "user_id": user_id,
            "reward": reward,
            "feedback_type": feedback_type,
            "timestamp": timestamp
        }
        try:
            with open(self.log_path, "r+") as f:
                data = json.load(f)
                data.append(feedback_entry)
                f.seek(0)
                json.dump(data, f)
                f.truncate()
        except Exception:
            pass

    def get_average_reward(self, user_id: str, window: int = 20) -> float:
        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)
            feedbacks = [i for i in data if i.get("user_id") == user_id and "reward" in i]
            if not feedbacks:
                return 0.0
            # Only consider the most recent 'window' feedbacks
            feedbacks = feedbacks[-window:]
            avg = sum(i["reward"] for i in feedbacks) / len(feedbacks)
            return avg
        except Exception:
            return 0.0

    def generate_report(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)
            now = datetime.now()
            filtered = [
                i for i in data
                if i.get("user_id") == user_id and
                   (now - datetime.fromisoformat(i.get("timestamp", now.isoformat()))).days <= days
            ]
            stats = {
                "total_interactions": len(filtered),
                "common_intents": {},
                "emotion_distribution": {},
                "active_hours": set()
            }
            for i in filtered:
                intent = i.get("intent", "general")
                stats["common_intents"][intent] = stats["common_intents"].get(intent, 0) + 1
                emotion = i.get("emotion", "neutral")
                stats["emotion_distribution"][emotion] = stats["emotion_distribution"].get(emotion, 0) + 1
                hour = datetime.fromisoformat(i.get("timestamp", now.isoformat())).hour
                stats["active_hours"].add(hour)
            stats["active_hours"] = sorted(list(stats["active_hours"]))
            return stats
        except Exception:
            return {}
