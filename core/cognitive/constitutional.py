import yaml
from typing import Dict, Any
import logging
logger = logging.getLogger(__name__)


class ConstitutionalGuard:
    def __init__(self):
        try:
            with open("config/constraints.yaml") as f:
                self.rules = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(
                "constraints.yaml not found. Using default constraints."
                )
            self.rules = {
                "safety": {"max_physical_actions": 5, "max_intensity": 0.7},
                "privacy": {"blacklist": ["ssn", "credit_card", "password"]}
            }
        except yaml.YAMLError as e:
            logger.error(
                f"Error parsing constraints.yaml: {e}. Using defaults."
                )
            self.rules = {
                "safety": {"max_physical_actions": 5, "max_intensity": 0.7},
                "privacy": {"blacklist": ["ssn", "credit_card", "password"]}
            }

    def validate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if action.get("type") == "physical":
            return self._check_physical_safety(action)
        if "web_access" in action:
            return self._check_privacy(action)
        return action

    def _check_physical_safety(self, action: Dict) -> Dict:
        if not isinstance(action, dict):
            logger.error("Action must be a dictionary")
            return {"type": "virtual", "reason": "Invalid action format"}
        max_actions = self.rules["safety"].get("max_physical_actions", 5)
        max_intensity = self.rules["safety"].get("max_intensity", 0.7)
        if (action.get("count", 0) >= max_actions or
                action.get("intensity", 0) > max_intensity):
            return {
                "type": "virtual",
                "reason": "Safety threshold exceeded",
                "original_action": action
            }
        return action

    def _check_privacy(self, action: Dict) -> Dict:
        sensitive_keywords = self.rules["privacy"].get("blacklist", [])
        query = action.get("query", "").lower()
        if any(keyword in query for keyword in sensitive_keywords):
            action["web_access"] = False
            action["response"] = "Cannot process sensitive requests"
        return action