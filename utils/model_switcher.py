# model_switcher.py
import yaml
import ollama
import logging
from typing import Optional
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the ModelSwitcher class
# This class is responsible for switching between models
# based on a configuration file.


class ModelSwitcher:
    async def switch(self, expected_hash: Optional[str] = None) -> bool:
        """Switch to a pending model if available and validated."""
        new_config = None
        with self.lock:
            try:
                # Load the model configuration
                with open(self.config_path, "r") as f:
                    config = yaml.safe_load(f)
                    pending_model = config["model"].get("pending")

                    if not pending_model:
                        self.logger.info("No pending model to switch to.")
                        return False

                    new_config = config.copy()

                # Validate the pending model by sending a test message
                test_response = await ollama.chat(
                    model=pending_model,
                    messages=[{"role": "user",
                               "content": "Can you acknowledge this test?"}]
                )

                # Check if response contains "acknowledge"
                if not test_response or "message" not in test_response:
                    self.logger.error(
                        (
                            f"Model {pending_model} failed validation - "
                            "No response"
                        )
                        )
                    return False

                message_content = test_response["message"].get(
                    "content", ""
                ).lower()
                if "acknowledge" not in message_content:
                    return False

                # Apply the model switch
                new_config["model"]["active"] = pending_model
                new_config["model"]["pending"] = None

                with open(self.config_path, "w") as f:
                    yaml.dump(new_config, f)

                self.logger.info(
                    f"Successfully switched to model: {pending_model}"
                    )
                return True

            except Exception as e:
                self.logger.error(f"Switch failed, reverting: {e}")

                # Rollback changes to prevent corrupting the config
                if new_config is None:
                    with open(self.config_path, "r") as f:
                        new_config = yaml.safe_load(f)

                new_config["model"]["pending"] = None
                with open(self.config_path, "w") as f:
                    yaml.dump(new_config, f)

                return False

    async def verify_model_availability(self, model_name: str) -> bool:
        try:
            available_models = await ollama.list()
            return any(
                m["name"] == model_name for m in available_models["models"]
            )
        except Exception as e:
            self.logger.error(f"Failed to verify model availability: {e}")
            return False
