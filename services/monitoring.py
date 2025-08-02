import psutil  # type: ignore
import torch
import asyncio
import logging
from prometheus_client import start_http_server
from prometheus_client.metrics_core import Gauge


class SystemMonitor:
    def __init__(self):
        self.metrics = {}
        try:
            start_http_server(8000)
            self.metrics["response_quality"] = Gauge(
                "response_quality", "User feedback score"
            )
            self.metrics["gpu_usage"] = Gauge("gpu_usage_bytes", "VRAM usage in bytes")
            self.metrics["cpu_usage"] = Gauge("cpu_usage", "CPU usage percentage")
            self.metrics["ram_usage"] = Gauge("ram_usage", "RAM usage percentage")
            # Add learning metrics
            self.metrics["learning_progress"] = Gauge(
                "learning_progress", "Model fine-tuning progress percentage"
            )
            self.metrics["adaptation_score"] = Gauge(
                "adaptation_score", "User-specific adaptation score"
            )
            self.metrics["interaction_count"] = Gauge(
                "interaction_count", "Total interactions processed"
            )
            self.metrics["personalization_level"] = Gauge(
                "personalization_level", "Level of personalization achieved"
            )
            logging.info("Prometheus metrics server started on port 8000")
        except Exception as e:
            logging.error(f"Error starting Prometheus server: {e}")

    def log_metric(self, name: str, value: float):
        if name in self.metrics:
            self.metrics[name].set(value)
        else:
            logging.warning(f"Attempted to log unknown metric: {name}")

    def track_performance(self):
        # Update GPU metrics if available
        if torch.cuda.is_available():
            self.metrics["gpu_usage"].set(torch.cuda.memory_allocated())
        
        # Update CPU and RAM metrics
        self.metrics["cpu_usage"].set(psutil.cpu_percent())
        self.metrics["ram_usage"].set(psutil.virtual_memory().percent)
        
    async def track_adaptation(self, user_id: str, personality_engine=None):
        """Track user-specific adaptation progress"""
        if not personality_engine:
            return
            
        try:
            # Get user profile to measure adaptation
            user_profile = personality_engine.user_profiles.get(user_id, {})
            interactions = user_profile.get("interactions", 0)
            
            # Log interaction count
            self.metrics["interaction_count"].set(interactions)
            
            # Calculate personalization level (0-100%)
            if interactions > 0:
                # Base level on number of emotional preferences learned
                emotional_prefs = len(user_profile.get("emotional_preferences", {}))
                # Max possible emotions from config
                max_emotions = 4  # happy, sad, angry, neutral as baseline
                
                # Calculate percentage (cap at 100%)
                personalization = min(100, (emotional_prefs / max_emotions * 100))
                
                # Factor in interaction count (increases with more interactions)
                interaction_factor = min(1, interactions / 100)  # Caps at 100 interactions
                
                # Combined adaptation score
                adaptation_score = personalization * interaction_factor
                
                self.metrics["adaptation_score"].set(adaptation_score)
                self.metrics["personalization_level"].set(personalization)
        except Exception as e:
            logging.error(f"Error tracking adaptation: {e}")
    
    async def run_diagnostics(self):
        """Run comprehensive system diagnostics and log metrics"""
        self.track_performance()
        
        # Check database size and update
        try:
            import os
            if os.path.exists("memory/context.db"):
                db_size = os.path.getsize("memory/context.db") / (1024 * 1024)  # MB
                self.log_metric("database_size_mb", db_size)
        except Exception as e:
            logging.error(f"Error checking database size: {e}")
            
        # Check model status
        try:
            if os.path.exists("models"):
                # Count number of fine-tuned models
                model_count = len([f for f in os.listdir("models") if f.startswith("jenna-ft-")])
                self.log_metric("finetuned_models", model_count)
        except Exception as e:
            logging.error(f"Error checking model status: {e}")


class EthicalMonitor:
    def __init__(self):
        self.constitution = self._load_constraints()
        
    def _load_constraints(self):
        try:
            import yaml
            with open("config/constraints.yaml") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load constraints: {e}")
            # Default fallback constraints
            return {
                "prohibited_content": ["harmful", "illegal", "unethical"],
                "safety": {"max_physical_actions": 5, "max_intensity": 0.7}
            }

    def validate_response(self, response: str) -> bool:
        return all(
            keyword not in response.lower()
            for keyword in self.constitution.get("prohibited_content", [])
        )


class HealthMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def check_system_health(self):
        """Check system components health"""
        checks = {
            "database": await self._check_db(),
            "audio": await self._check_audio(),
            "model": await self._check_model(),
            "memory": await self._check_memory()
        }
        return all(checks.values()), checks
    
    async def _check_db(self):
        """Check database connectivity and integrity"""
        try:
            # Import here to avoid circular imports
            from core.memory.context_db import ContextDatabase
            db = ContextDatabase()
            # Simple check - see if we can get a connection
            async with db.get_connection() as conn:
                await conn.execute("PRAGMA integrity_check")
                return True
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False
    
    async def _check_audio(self):
        """Check audio system functionality"""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            # Check if we have at least one input and one output device
            has_input = any(device['max_input_channels'] > 0 for device in devices)
            has_output = any(device['max_output_channels'] > 0 for device in devices)
            return has_input and has_output
        except Exception as e:
            self.logger.error(f"Audio health check failed: {e}")
            return False
    
    async def _check_model(self):
        """Check model availability and functioning"""
        try:
            import ollama
            # Try a simple model query
            response = await asyncio.to_thread(
                ollama.chat,
                model="mistral",
                messages=[{"role": "user", "content": "Test"}]
            )
            return bool(response and "message" in response)
        except Exception as e:
            self.logger.error(f"Model health check failed: {e}")
            return False
    
    async def _check_memory(self):
        """Check memory system functionality"""
        try:
            # Check available system memory
            free_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
            if free_memory < 1.0:  # Less than 1GB free
                self.logger.warning(f"Low memory available: {free_memory:.2f}GB")
                return False
                
            # Check if vector storage is accessible
            from core.memory.vector_store import VectorMemory
            memory = VectorMemory()
            # Simple check - retrieve something to verify connectivity
            test_results = memory.retrieve("test query", n=1)
            return True
        except Exception as e:
            self.logger.error(f"Memory health check failed: {e}")
            return False
