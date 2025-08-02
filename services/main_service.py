import asyncio
import logging
from utils.config_loader import get_config, cached_config
from bootstrap import setup_environment
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import os
from core.cognitive.emotion import EmotionAnalyzer
import sounddevice as sd
import torch

# Importing necessary modules for JENNA AI system

# Core cognitive components
from core.cognitive.reasoner import AdvancedReasoner
from core.cognitive.intent_classifier import IntentClassifier

# Perception modules
from core.perception.audio_service import AudioProcessor
from core.perception.wake_word import WakeWordDetector

# Memory systems
from core.memory.vector_store import VectorMemory
from core.memory.context_db import ContextDatabase
from core.integration.skills.workflow_engine import WorkflowEngine

# Training and learning
from training.self_learner import SelfLearningEngine
from training.data_prep import DataPreparer

# Integration and services
from core.integration.skills.web_ops import WebOperator
from core.integration.skills.api_integration import APIManager
from services.security_service import SecurityMonitor
from services.monitoring import SystemMonitor
from utils.model_switcher import ModelSwitcher
from core.personality.habit_tracker import SmartHabitAI, ProactiveSuggestor
from monitoring.ethics import EthicsGuardian
from core.personality.adaptive import PersonalityEngine
from core.personality.humor_engine import HumorEngine
from core.analytics.analytics_engine import AnalyticsEngine
from core.learning.unstructured_learner import UnstructuredLearner
from core.knowledge.instant_acquirer import InstantKnowledgeAcquirer
from core.self_repair.repair_engine import SelfRepairEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

setup_environment()

class JENNA:
    """
    JENNA: Just Evolved Neural Network Assistant
    A self-learning AI with natural interactions and improving over time.
    """

    def __init__(self):
        """Initialize the JENNA AI system with all required components."""
        logger.info("Initializing JENNA AI system...")

        # Load configuration
        self.model_config = get_config("model.yaml")
        self.model_name = self._load_model_config()

        # System state
        self.session_id = "default_session"
        self.user_id = "default_user"
        self.direct_mode = False
        self.shutdown_event = asyncio.Event()
        self.initialized = False
        self.mode_change_callback = None  # Notify other components of mode changes

        # Initialize core components
        self._init_core_components()

        logger.info("JENNA AI core initialized, preparing for activation")
        print(" JENNA AI Activated! Ready for commands.")

    def _verify_configs(self):
        required_files = [
            # configs are now validated and loaded via config_loader
            # Example usage: get_config("model.yaml")
            "model.yaml",
            "constraints.yaml",
            "personality.yaml",
            "workflows.yaml",
            "api_urls.yaml"
        ]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Missing required config: {file}")

    async def verify_dependencies(self):
        """Verify all required dependencies"""
        try:
            # Check audio devices
            devices = sd.query_devices()
            if not devices:
                logger.error("No audio devices found. Voice interaction will be unavailable.")
                return False
                
            # Check for at least one input device (microphone)
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                logger.error("No microphone found. Voice input will be unavailable.")
                
            # Check for at least one output device (speakers)
            output_devices = [d for d in devices if d['max_output_channels'] > 0]
            if not output_devices:
                logger.error("No speakers found. Voice output will be unavailable.")

            # Check GPU if available
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_properties(0)
                logger.info(f"GPU detected: {gpu_info.name} with {gpu_info.total_memory/1e9:.2f}GB memory")
                torch.cuda.memory_summary(device=0, abbreviated=True)
            else:
                logger.info("Running in CPU mode")

            return True
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return False

    def _load_model_config(self) -> str:
        """Load model configuration from YAML file (now via config_loader)."""
        try:
            # Use config loaded during initialization
            return self.model_config["model"]["active"]
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            return "default_model"

    def _init_core_components(self):
        """Initialize all core components of the system."""
        # Cognitive components
        self.reasoner = AdvancedReasoner()
        self.intent_classifier = IntentClassifier()
        self.workflow_engine = WorkflowEngine()

        # Perception components
        self.audio = AudioProcessor()
        # Pass a callback to WakeWordDetector for integration
        self.wake_detector = WakeWordDetector(self._on_wake_word_detected)

        # Memory systems
        self.memory = VectorMemory()
        self.context_db = ContextDatabase()

        # Training and learning
        self.learner = SelfLearningEngine()
        self.data_prep = DataPreparer()
        self.model_switcher = ModelSwitcher()

        # Integration and services
        self.web_operator = WebOperator()
        self.api_manager = APIManager()
        self.security = SecurityMonitor()
        self.monitor = SystemMonitor()
        # Initialize EmotionAnalyzer for emotional intelligence
        self.emotion_analyzer = EmotionAnalyzer(self.model_name)
        # self.self_finetuner = SelfFineTuner()  # Removed invalid reference
        self.habit_ai = SmartHabitAI()
        self.suggestor = ProactiveSuggestor()
        self.ethics_guardian = EthicsGuardian()
        self.personality = PersonalityEngine()
        
        # Humor, analytics, unstructured learning, instant knowledge, self-repair
        self.humor_engine = HumorEngine()
        self.analytics_engine = AnalyticsEngine()
        self.unstructured_learner = UnstructuredLearner()
        self.instant_knowledge = InstantKnowledgeAcquirer()
        self.self_repair = SelfRepairEngine()
        # Last suggestion time tracking
        self.last_suggestion_time = datetime.now().timestamp()
        self.suggestion_cooldown = 3600  # 1 hour between suggestions

    async def initialize(self):
        """Perform asynchronous initialization of all subsystems."""
        if self.initialized:
            return

        logger.info("Initializing JENNA subsystems...")

        try:
            # Check dependencies first
            if not await self.verify_dependencies():
                logger.warning("Some dependencies not available - functionality may be limited")

            # Initialize database and memory systems first
            await self.context_db._initialize()
            
            # Initialize wake word detector
            try:
                await self.wake_detector.initialize()
                logger.info("Wake word detector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize wake word detector: {e}")

            # Start concurrent initialization of remaining subsystems
            await asyncio.gather(
                self._init_memory_systems(),
                self._load_models(),
                self.web_operator.browser.setup(),
            )

            self.initialized = True
            logger.info("JENNA subsystems initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize subsystems: {e}")
            await self.audio.synthesize(
                "System initialization failed. Some functions may be limited."
            )

    async def _init_memory_systems(self):
        """Initialize and prepare memory systems for the session."""
        try:
            logger.info(f"Initializing memory systems for session: {self.session_id}")
            
            # Create directory structure if it doesn't exist
            Path("memory").mkdir(exist_ok=True)
            
            # Check if session exists or create it
            session_exists = await self.context_db.session_exists(self.session_id)
            if not session_exists:
                logger.info(f"Creating new session: {self.session_id}")
                await self.context_db.create_session(self.session_id, self.user_id)
            
            # Load recent conversations for context
            recent_convos = await self.context_db.retrieve_recent_conversations(
                self.session_id, limit=5
            )
            if recent_convos:
                logger.info(f"Loaded {len(recent_convos)} recent conversations")
                
            # Store the recent conversations in the reasoner's context
            for convo in recent_convos:
                await self.reasoner.update_context_cache(
                    self.session_id, 
                    convo.get("input", ""), 
                    convo.get("response", "")
                )
                
            # Initialize habits tracking system
            try:
                # Check if habit_ai has initialize method (not all implementations might have this)
                if hasattr(self.habit_ai, "initialize"):
                    await self.habit_ai.initialize(self.user_id)
                    logger.info("Habit AI initialized")
            except Exception as e:
                logger.error(f"Error initializing habit AI: {e}")
                
            logger.info(f"Memory systems prepared for session: {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Memory initialization failed: {e}")
            return False

    async def _cleanup_resources(self):
        """Clean up system resources"""
        try:
            await asyncio.gather(
                self.context_db.close(),
                self.audio.cleanup_resources(),
                self.web_operator.browser.cleanup()
            )
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def _load_models(self):
        """Load and prepare ML models required by the system."""
        logger.info("Loading ML models...")
        try:
            # Preload the reasoning model
            preload_response = await self.reasoner.preload_model()
            
            # Load emotion model
            emotion_status = await self.emotion.initialize()
            
            # Load additional models as needed
            await self.intent_classifier.load_model()
            
            # Register for model update notifications
            asyncio.create_task(self.reasoner.monitor_model_updates())
            
            # Schedule background tasks
            asyncio.create_task(self.schedule_finetuning())
            asyncio.create_task(self.ethics_guardian.periodic_audit())
            
            logger.info(" All models successfully loaded")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    async def schedule_finetuning(self):
        """Periodically check if finetuning should be performed.
        This runs as a background task to ensure continuous improvement."""
        try:
            while not self.shutdown_event.is_set():
                # Check daily if sufficient new data is available
                await asyncio.sleep(86400)  # 24 hours
                
                if await self._should_finetune():
                    logger.info("Scheduled finetuning triggered")
                    await self._perform_finetuning()
                else:
                    logger.info("Not enough new data for scheduled finetuning")
                    
        except asyncio.CancelledError:
            logger.info("Finetuning scheduler cancelled")
        except Exception as e:
            logger.error(f"Error in finetuning scheduler: {e}")

    async def _check_proactive_suggestions(self):
        """Check if it's appropriate to offer a proactive suggestion."""
        current_time = datetime.now().timestamp()
        
        # Only suggest if enough time has passed since last suggestion
        if current_time - self.last_suggestion_time < self.suggestion_cooldown:
            return
            
        try:
            # Generate personalized suggestions
            suggestions = await self.suggestor.generate_suggestions(self.user_id)
            
            if suggestions and len(suggestions) > 0:
                suggestion = suggestions[0]
                # Only suggest if we have a meaningful suggestion
                if suggestion and len(suggestion.get('message', '')) > 20:
                    await self._handle_output({
                        "text": f"I noticed something that might interest you: {suggestion['message']}",
                        "emotion": "helpful",
                        "is_proactive": True
                    })
                    self.last_suggestion_time = current_time
                    logger.info(f"Offered proactive suggestion: {suggestion['type']}")
        except Exception as e:
            logger.error(f"Error generating proactive suggestions: {e}")

    async def run(self):
        """
        Main run loop for JENNA. This method handles initialization,
        processing inputs, and cleanup on shutdown.
        """
        try:
            # Initialize all subsystems
            await self.initialize()
            
            # Start wake word detection in a separate task
            wake_detector_task = None
            if not self.direct_mode:
                wake_detector_task = asyncio.create_task(
                    self.wake_detector.start()
                )
            
            logger.info("JENNA is now running and ready for interactions")
            
            # Run until shutdown is requested
            while not self.shutdown_event.is_set():
                try:
                    # Periodically check for proactive suggestions
                    if not self.direct_mode:
                        await self._check_proactive_suggestions()
                    
                    # Wait a bit before checking again
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    logger.info("Run loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in run loop: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
        
        except Exception as e:
            logger.error(f"Critical error in run loop: {e}")
        finally:
            # Cancel the wake detector task if it exists
            if wake_detector_task and not wake_detector_task.done():
                wake_detector_task.cancel()
                try:
                    await wake_detector_task
                except asyncio.CancelledError:
                    pass
                
            # Clean up resources
            await self._cleanup_resources()
            logger.info("JENNA has shut down")

    async def _on_wake_word_detected(self, detected_text: str):
        """Handle wake word detection by starting a conversation."""
        self.logger.info(f"Wake word detected: {detected_text}")
        
        try:
            # Record audio for the user's command
            await self.audio.speak_text("Yes?")
            audio_data = await self.audio.record_audio_async(5)  # Record 5 seconds
            
            # Process the audio command
            transcription = await self.audio.transcribe(audio_data)
            if transcription and "text" in transcription:
                text = transcription["text"]
                self.logger.info(f"Transcribed: {text}")
                
                # Process the command and get a response
                response = await self._handle_interaction({
                    "text": text,
                    "audio_context": transcription.get("prosody", {})
                })
                
                # Speak the response
                await self.audio.speak_text(response["text"], response.get("emotion", "neutral"))
        except Exception as e:
            self.logger.error(f"Error processing wake word interaction: {e}")
            await self.audio.speak_text("I'm sorry, I couldn't process that request.")

    async def _handle_interaction(self, input_data: Dict) -> Dict:
        """Process user interaction and generate a response.
        
        Args:
            input_data: Dictionary containing text and optional audio context
            
        Returns:
            Response dictionary with text and metadata
        """
        try:
            if not input_data or not input_data.get("text"):
                return {"error": "No input provided", "text": "I didn't catch that"}
            
            # Get text and optional audio context
            text = input_data["text"]
            audio_context = input_data.get("audio_context", {})
            
            # Process through reasoner
            response = await self.reasoner.process_query(
                text, audio_context, self.session_id
            )
            
            # Store interaction in memory systems
            await self._store_interaction(text, response)
            
            # Update user's habit tracking
            self.habit_ai.log_interaction(
                self.user_id, 
                response.get("intent", "query"), 
                response.get("entities", [])
            )
            
            # Log the interaction for learning
            self.learner.log_interaction({
                "input": text,
                "response": response.get("text", ""),
                "session_id": self.session_id,
                "user_id": self.user_id,
                "emotion": response.get("emotion", "neutral"),
                "intent": response.get("intent", "query")
            })
            
            return response
        except Exception as e:
            logger.error(f"Error handling interaction: {e}")
            return {
                "text": "I encountered an issue processing your request.",
                "error": str(e)
            }

    async def _store_interaction(self, input_text: str, response: Dict):
        """Store interaction in both context DB and vector memory.
        
        Args:
            input_text: User input text
            response: Response dictionary
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Store in context database with full metadata
            metadata = {
                "emotion": response.get("emotion", "neutral"),
                "intent": response.get("intent", "query"),
                "timestamp": timestamp,
                "session_id": self.session_id,
                "user_id": self.user_id,
                "entities": response.get("entities", [])
            }
            
            # Ensure we're storing text from the response
            response_text = response.get("text", "")
            if not response_text:
                logger.warning("Empty response text in _store_interaction")
                response_text = "No response generated"
            
            # Log full interaction in context database
            await self.context_db.log_interaction(
                self.session_id,
                input_text,
                response_text,
                metadata
            )
            
            # Generate embedding for vector-based retrieval
            combined_text = f"User: {input_text}\nJENNA: {response_text}"
            vector_metadata = {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "timestamp": timestamp,
                "type": "conversation"
            }
            
            # Store in vector memory for semantic search
            self.memory.store(
                combined_text,
                vector_metadata
            )
            
            logger.debug(f"Interaction stored: {input_text[:30]}...")
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")

if __name__ == "__main__":
    jenna = JENNA()
    try:
        asyncio.run(jenna.run())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        logger.info("JENNA AI system has shut down.")
