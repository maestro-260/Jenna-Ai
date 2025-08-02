import ollama
from utils.config_loader import get_config, cached_config
import asyncio
import torch
import httpx
import logging
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from core.integration.skills.workflow_engine import WorkflowEngine
from core.cognitive.cognitive_controller import CognitiveController
from core.cognitive.tree_of_thought import TreeOfThoughtEngine
from core.cognitive.commonsense import CommonsenseReasoner
from core.cognitive.planner import CognitivePlanner
from core.cognitive.reflection import SelfReflectionEngine

logger = logging.getLogger(__name__)

# Lazy imports will be handled in _lazy_import_dependencies()


class AdvancedReasoner:
    """Core reasoning module that coordinates cognitive functions."""

    def __init__(self):
        """Initialize the reasoning module with default configuration."""
        self._lazy_import_dependencies()

        try:
            model_cfg = get_config("model.yaml")
            self.llm = model_cfg["model"]["active"]
        except Exception as e:
            logger.error(f"Error loading model config: {e}")
            # Fallback to default model
            self.llm = "mistral"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.memory = VectorMemory()
        self.context_db = ContextDatabase()
        self.emotion = EmotionAnalyzer(active_model=self.llm)
        self.safety = ConstitutionalGuard()
        self.intent_classifier = IntentClassifier()
        self.model_switcher = ModelSwitcher()
        self.api_manager = APIManager()
        self.workflow_engine = WorkflowEngine()

        self.params = {
            'temp': 0.7,
            'empathy': 1.0,
            'verbosity': 0.5,
            'creativity': 0.3
        }

        self.context_cache = {}
        self.shutdown_event = asyncio.Event()  # Add shutdown event

        # Initialize cognitive architecture components
        try:
            self.cognitive_controller = CognitiveController(active_model=self.llm)
            self.tree_of_thought = TreeOfThoughtEngine(active_model=self.llm)
            self.commonsense = CommonsenseReasoner(active_model=self.llm)
            self.planner = CognitivePlanner(active_model=self.llm)
            self.reflection = SelfReflectionEngine(active_model=self.llm)
            
            # Flag for advanced cognitive capabilities
            self.use_cognitive_architecture = True
            
            logger.info("Advanced cognitive architecture initialized")
        except Exception as e:
            logger.error(f"Error initializing cognitive architecture: {e}")
            self.use_cognitive_architecture = False

    def _lazy_import_dependencies(self):
        """Import dependencies lazily to avoid circular imports."""
        global EmotionAnalyzer, ConstitutionalGuard, \
            ContextDatabase, VectorMemory, IntentClassifier, \
            ModelSwitcher, APIManager

        from core.cognitive.emotion import EmotionAnalyzer
        from core.cognitive.constitutional import ConstitutionalGuard
        from core.memory.context_db import ContextDatabase
        from core.memory.vector_store import VectorMemory
        from core.cognitive.intent_classifier import IntentClassifier
        from utils.model_switcher import ModelSwitcher
        from core.personality.habit_tracker import SmartHabitAI
        from core.integration.skills.api_integration import APIManager

        self.habit_ai = SmartHabitAI()

    async def monitor_model_updates(self):
        """Periodically check for model updates."""
        while not self.shutdown_event.is_set():
            await asyncio.sleep(3600)
            if await self.model_switcher.switch(None):
                await self.reload_components()
                logger.info(f"Switched to new model: {self.llm}")

    async def stop_monitoring(self):
        """Stop the model update monitoring loop."""
        self.shutdown_event.set()

    async def preload_model(self):
        """Preload the LLM to minimize first-request latency."""
        try:
            # This creates a simple exchange to initialize the model
            logger.info(f"Preloading model: {self.llm}")
            initial_prompt = "Hello, JENNA system initialization."
            
            # Use a timeout to prevent hanging during initialization
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    ollama.chat,
                    model=self.llm,
                    messages=[{"role": "user", "content": initial_prompt}]
                ),
                timeout=30
            )
            
            logger.info(f"Model preloaded successfully: {self.llm}")
            return {"status": "success", "model": self.llm}
        except asyncio.TimeoutError:
            logger.error(f"Timeout while preloading model {self.llm}")
            return {"status": "error", "reason": "timeout"}
        except Exception as e:
            logger.error(f"Error preloading model {self.llm}: {e}")
            return {"status": "error", "reason": str(e)}
    
    async def handle_user_input(
            self, text: str, session_id: str
            ) -> Dict[str, Any]:
        """
        Handler for user input to intelligently route btwn workflows and LLM.
        
        Args:
            text: The user's text input
            session_id: The session identifier
            
        Returns:
            A response dictionary with text and metadata
        """
        # First, classify intent with confidence score
        intent_result = await self.intent_classifier.classify(text)
        intent = intent_result.get("intent", "general_query")
        confidence = intent_result.get("confidence", 0.0)
        
        # Check if have user context that might help with intent disambiguation
        user_context = await self._get_user_context(session_id)
        
        # Store input for context tracking
        await self.update_context_cache(session_id, text, "")
        
        # Track workflow execution for analytics
        workflow_executed = False
        workflow_result = None
        
        # If intent matches a workflow with sufficient confidence, execute it
        if intent in self.workflow_engine.workflows and confidence > 0.65:
            try:
                logger.info(
                    f"Executing workflow '{intent}' for input: {text}"
                    )
                workflow_result = await self.workflow_engine.execute_workflow(
                    intent, {
                        "text": text,
                        "session_id": session_id,
                        "user_context": user_context
                    })
                
                # Check if workflow executed successfully
                if "error" not in workflow_result:
                    workflow_executed = True
                    
                    # Store workflow result in context for future reference
                    await self.update_context_cache(
                        session_id, 
                        f"_WORKFLOW_{intent}",  # Special marker in context
                        str(workflow_result)
                    )
                    
                    # Format workflow result into natural language response
                    if isinstance(
                            workflow_result, dict
                            ) and "response_text" in workflow_result:
                        response_text = workflow_result["response_text"]
                    else:
                        # Generate natural language from structured workflow 
                        response_text = await self._format_workflow_result(
                            intent, workflow_result
                            )
                        
                    return {
                        "text": response_text,
                        "workflow_executed": True,
                        "intent": intent,
                        "workflow_result": workflow_result
                    }
                else:
                    # Log workflow execution error
                    logger.warning(
                        f"Workflow '{intent}' execution failed: "
                        f"{workflow_result['error']}"
                    )
            except Exception as e:
                logger.error(f"Error executing workflow '{intent}': {e}")
        
        # If workflow wasn't executed or failed, fall back to LLM reasoning
        if not workflow_executed:
            # For low confidence matches or workflow failures,
            #  include that context in reasoning
            extra_context = {}
            if intent in self.workflow_engine.workflows and confidence <= 0.65:
                extra_context["potential_intent"] = intent
                extra_context["intent_confidence"] = confidence
            elif workflow_result and "error" in workflow_result:
                extra_context["workflow_error"] = workflow_result["error"]
                
            # Generate response using LLM chain-of-thought
            llm_response = await self._generate_llm_response(
                text, session_id, extra_context
                )
            
            # Update context with response
            await self.update_context_cache(
                session_id, text, llm_response["text"]
                )
            
            return llm_response

    async def _get_user_context(self, session_id: str) -> Dict[str, Any]:
        """Get relevant user context to help with intent disambiguation."""
        try:
            # Get recent interactions from context cache
            recent_interactions = self.context_cache.get(session_id, [])
            
            # Get user preferences from context database
            user_id = await self.context_db.get_user_id(session_id)
            user_prefs = await self.context_db.get_user_preferences(
                user_id
                ) if user_id else {}
            
            return {
                "recent_interactions": (
                    recent_interactions[-5:] if recent_interactions else []
                ),
                "preferences": user_prefs,
                "session_id": session_id
            }
        except Exception as e:
            logger.error(f"Error retrieving user context: {e}")
            return {"session_id": session_id}

    async def _format_workflow_result(self, intent: str, result: Any) -> str:
        """Generate natural language ouput from structured workflow result."""
        # For simple string results, just return them
        if isinstance(result, str):
            return result
            
        # Convert dict to json for LLM formatting
        import json
        result_json = json.dumps(result)
        
        prompt = f"""
        Format this structured result from a '{intent}' workflow into.
        a natural, conversational response:
        
        {result_json}
        
        Your response should be helpful, concise,
        and directly address the user's needs.
        Include only the final response text,
        no explanations or meta-commentary.
        """
        
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.llm,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            logger.warning(f"Error formatting workflow result: {e}")
            # Fallback to simple string conversion if formatting fails
            return f"Here's what I found: {str(result)}"

    async def _generate_llm_response(
            self,
            text: str,
            session_id: str,
            extra_context: Optional[Dict] = None
            ) -> Dict[str, Any]:
        """Generate a response using LLM chain-of-thought reasoning."""
        if extra_context is None:
            extra_context = {}
            
        # Get analysis of input
        analysis = await self._analyze_input(text, {})
        
        # Include any workflow context in the analysis
        for key, value in extra_context.items():
            analysis[key] = value
        
        # Generate response using chain-of-thought reasoning
        response_text = await self._generate_response(
            text, analysis, session_id
            )
        
        # Format the output
        return self._format_output(response_text, analysis)

    async def check_proactive_suggestions(self, session_id: str) -> List[str]:
        """Check if there are any proactive suggestions to offer"""
        # If we have commonsense reasoner available, use it
        if self.use_cognitive_architecture:
            try:
                # Get user context
                user_context = await self._get_user_context(session_id)
                
                # Get recent conversations
                recent_convos = user_context.get("recent_conversations", [])
                recent_texts = [conv.get("text", "") for conv in recent_convos]
                combined_text = " ".join(recent_texts)
                
                # Use commonsense reasoning to infer relevant information
                inferred_facts = await self.commonsense.infer_facts(combined_text)
                
                # Generate suggestions based on facts
                suggestions = []
                for fact in inferred_facts[:2]:  # Limit to top 2 facts
                    suggestions.append(f"Based on our conversation, you might want to consider: {fact}")
                
                if suggestions:
                    return suggestions
                # Fall back to standard approach below
            except Exception as e:
                logger.error(f"Error generating proactive suggestions: {e}")
                # Fall back to standard approach below
        
        # Original proactive suggestions logic
        # ... existing code continues ...

    async def process_query(
            self,
            text: str,
            audio_context: Optional[Dict] = None,
            session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process a user query with full reasoning capabilities
        
        Args:
            text: The user's text query
            audio_context: Optional audio context information
            session_id: The user's session ID
            
        Returns:
            Response dictionary with reasoning details
        """
        # Analyze input to determine intent, emotion, etc.
        analysis = await self._analyze_input(text, audio_context or {})
        
        # Get user context from session
        user_context = await self._get_user_context(session_id)
        
        # Combine contexts
        combined_context = {
            **user_context,
            **analysis,
            "session_id": session_id
        }
        
        # If we have cognitive architecture available, use it
        if self.use_cognitive_architecture:
            try:
                # Use the cognitive controller to select and apply the best approach
                cognitive_result = await self.cognitive_controller.process_query(text, combined_context)
                
                # Extract response text
                response_text = cognitive_result.get("text", "")
                
                # Format the output
                result = self._format_output(response_text, analysis)
                
                # Add cognitive details for debugging/monitoring
                result["cognitive_approach"] = cognitive_result.get("reasoning_approach", "unknown")
                
                # Update context with this interaction
                await self.update_context_cache(session_id, text, response_text)
                
                return result
                
            except Exception as e:
                logger.error(f"Error in cognitive processing: {e}")
                # Fall back to standard processing below
        
        # Standard (fallback) processing
        response_text = await self._generate_response(text, analysis, session_id)
        result = self._format_output(response_text, analysis)
        
        # Update context with this interaction
        await self.update_context_cache(session_id, text, response_text)
        
        return result

    async def _analyze_input(
            self, text: str, audio_context: Dict
            ) -> Dict[str, Any]:
        """Analyze input text and audio features.

        Args:
            text: User input text
            audio_context: Audio context with prosody features

        Returns:
            Analysis results dictionary
        """
        emotion_task = self.emotion.analyze(text, audio_context)
        safety_task = asyncio.to_thread(
            self.safety.validate_action, {"query": text}
        )
        context_task = asyncio.to_thread(self.memory.retrieve, text, n=3)

        emotion_detected, safety_result, context_results = (
            await asyncio.gather(
                emotion_task,
                safety_task,
                context_task
            )
        )

        emotional_reply = await self.emotion.generate_empathetic_reply(
            text, audio_context, emotion_detected
        )

        threat_level = safety_result.get("response", "safe")

        return {
            'text': text,
            'emotion': emotion_detected,
            'emotional_reply': emotional_reply,
            'threat_level': threat_level,
            'context': context_results,
            'audio_context': audio_context
        }

    async def _generate_response(
            self, text: str, analysis: Dict, session_id: str
            ) -> str:
        """Generate a response using chain-of-thought reasoning."""
        # Build the initial thought chain
        thoughts = await self._build_thought_chain(text, analysis)

        # Generate introspective analysis
        reflection = await self._introspect_thoughts(thoughts)

        # Construct final response incorporating thoughts
        prompt = self._build_dynamic_prompt(thoughts, reflection, analysis)

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": self.llm,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a dynamic AI capable"
                                               "of deep and nuanced thinking."
                                               "Express your thought naturally"
                                               " and "
                                               "build on them step by step."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            "temperature": 0.8,
                            # Increased for more creative responses
                            "top_p": 0.9
                        }
                    )
                    result = response.json()
                    return result["message"]["content"]
            except Exception as e:
                logger.error(
                    f"Generation error on attempt {attempt + 1}: {e}"
                    )
        return "I need a moment to collect my thoughts..."

    async def _build_thought_chain(
            self, text: str, analysis: Dict
            ) -> List[str]:
        """Build a chain of thoughts for reasoning"""
        # If we have the tree of thought engine available, use it
        if self.use_cognitive_architecture:
            try:
                context = {"analysis": analysis}
                tot_result = await self.tree_of_thought.solve(text, context)
                
                # Extract all thoughts from the branches
                thoughts = []
                for branch in tot_result.get("branches", []):
                    thoughts.append(branch["thought"])
                    # Add sub-thoughts from each branch
                    for sub_branch in branch.get("sub_thoughts", []):
                        if "thought" in sub_branch:
                            thoughts.append(f"  - {sub_branch['thought']}")
                
                # If we found thoughts, return them
                if thoughts:
                    return thoughts
                # Otherwise fall back to standard approach below
            except Exception as e:
                logger.error(f"Error using tree of thought: {e}")
                # Fall back to standard approach below
        
        # Original thought chain building logic
        prompt = (
            f"Question: {text}\n\n"
            "Think through this step by step:"
        )
        # ... existing code continues ...

    async def _introspect_thoughts(self, thoughts: List[str]) -> str:
        """Introspect on the thoughts to form a cohesive understanding"""
        # If we have the reflection engine available, use it
        if self.use_cognitive_architecture:
            try:
                # Create a reflection context
                thoughts_text = "\n".join(thoughts)
                reflection_data = {
                    "user_input": "Think about these thoughts",
                    "ai_response": thoughts_text
                }
                
                # Use the reflection engine
                reflection = await self.reflection.reflect_on_response(
                    reflection_data["user_input"],
                    reflection_data["ai_response"]
                )
                
                # Extract improvements
                improvements = reflection.get("improvements", [])
                if improvements:
                    return "\n".join(improvements)
                # Fall back to standard approach below
            except Exception as e:
                logger.error(f"Error in reflection: {e}")
                # Fall back to standard approach below
        
        # Original introspection logic
        thoughts_text = "\n".join(thoughts)
        # ... existing code continues ...

    def _build_dynamic_prompt(
            self, thoughts: List[str], reflection: str, analysis: Dict
            ) -> str:
        """Build a dynamic prompt incorporating the thought chain."""
        # Add workflow-specific considerations if present
        workflow_context = ""
        if "potential_intent" in analysis:
            workflow_context = f"""
            This query might relate to the "
            f"'{analysis['potential_intent']}' workflow.
            Consider if we should guide the user toward using "
            "this functionality explicitly.
            """
        elif "workflow_error" in analysis:
            workflow_context = f"""
            A workflow attempt failed with error: "
            f"{analysis['workflow_error']}
            Consider if we should explain this to the user or "
            "provide an alternative solution.
            """

        return f"""I've been thinking about this request:

Initial Context:
{analysis['text']}

{workflow_context}

My thought process:
{chr(10).join(thoughts)}

Upon reflection:
{reflection}

Given all of this, I want to respond in a way that:
1. Shows I've deeply considered the matter
2. Addresses both explicit and implicit aspects
3. Maintains emotional awareness
4. Provides genuine value

Help me formulate such a response, while keeping my own perspective and voice.
"""

    def _format_output(
            self, response_text: str, analysis: Dict
            ) -> Dict[str, Any]:
        """Format the final response with metadata.

        Args:
            response_text: The generated response text
            analysis: Analysis results dictionary

        Returns:
            Formatted response dictionary
        """
        return {
            "text": response_text,
            "emotion": analysis.get('emotion', 'neutral'),
            "original_input": analysis.get('text', ''),
            "intent": analysis.get('intent', 'general_query'),
            "entities": analysis.get('entities', []),
            "threat_level": analysis.get('threat_level', 'safe')
        }

    async def handle_confirmation(
            self, confirmation_text: str, original_intent: str, session_id: str
    ) -> Dict[str, Any]:
        """Handle user confirmation for sensitive actions.

        Args:
            confirmation_text: User's confirmation response
            original_intent: The original intent that required confirmation
            session_id: Session identifier

        Returns:
            Response dictionary
        """
        confirmation_lower = confirmation_text.lower()
        affirmative_words = {
            "yes", "yeah", "yep", "correct", "proceed",
            "confirm", "right", "ok", "okay",    # fix missing comma
            "affirmative", "sure", "go ahead",
            "absolutely", "definitely", "certainly",
            "indeed", "yes please", "by all means",
            "for sure", "yup", "absolutely"}
        negative_words = {
            "no", "nope", "don't", "stop", "cancel",
            "negative", "incorrect", "not sure", "not really",
            "no way", "never", "not at all", "not interested",
            "not", "wrong", "halt", "abort", "decline"}

        is_affirmative = any(
            word in confirmation_lower for word in affirmative_words
            )
        is_negative = any(
            word in confirmation_lower for word in negative_words
            )

        if is_affirmative and not is_negative:
            # If confirmed, execute the corresponding workflow
            try:
                workflow_result = await self.workflow_engine.execute_workflow(
                    original_intent, {
                        "text": confirmation_text,
                        "session_id": session_id,
                        "confirmed": True
                    })
                
                # Format response based on workflow result
                if "error" in workflow_result:
                    return {
                        "text": (
                            f"I encountered an issue: "
                            f"{workflow_result['error']}"
                        ),
                        "action_executed": False,
                    }
                    
                # Format workflow result into natural language
                response_text = await self._format_workflow_result(
                    original_intent, workflow_result
                    )
                return {
                    "text": response_text,
                    "action_executed": True,
                    "workflow_result": workflow_result
                }
            except Exception as e:
                logger.error(
                    f"Error executing confirmed workflow "
                    f"'{original_intent}': {e}"
                    )
                return {
                    "text": "I encountered an issue while processing."
                    "Please try again.",
                    "action_executed": False
                }
        else:
            return {
                "text": "I've cancelled that request.",
                "action_executed": False
                }

    async def update_context_cache(
            self, session_id: str, input_text: str, output_text: str
            ):
        """Update the context cache with a new interaction.

        Args:
            session_id: Session identifier
            input_text: User input text
            output_text: Assistant response text
        """
        if session_id not in self.context_cache:
            self.context_cache[session_id] = []

        self.context_cache[session_id].append({
            "input": input_text,
            "response": output_text,
            "timestamp": asyncio.get_event_loop().time()
        })

        self.context_cache[session_id] = self.context_cache[session_id][-10:]

    async def reload_components(self):
        """Reload components after model changes."""
        try:
            model_cfg = get_config("model.yaml")
            self.llm = model_cfg["model"]["active"]
            
            self.memory.collection = self.memory.client.get_collection(
                "context"
            )
            await asyncio.gather(
                self.emotion.analyze("warmup"),
                self.intent_classifier.classify("warmup")
            )

            logger.info(f"Components reloaded with model: {self.llm}")
        except Exception as e:
            logger.error(f"Error reloading components: {e}")