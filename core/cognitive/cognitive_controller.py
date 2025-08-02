import logging
from typing import Dict, List, Any, Optional
import asyncio
import time
import json

from core.cognitive.planner import CognitivePlanner
from core.cognitive.reflection import SelfReflectionEngine
from core.cognitive.tree_of_thought import TreeOfThoughtEngine
from core.cognitive.commonsense import CommonsenseReasoner

logger = logging.getLogger(__name__)

class CognitiveController:
    """
    Coordinates different cognitive components to achieve more human-like intelligence.
    This enables the AI to:
    1. Select the appropriate reasoning method for different tasks
    2. Combine insights from different reasoning approaches
    3. Monitor and improve cognitive performance
    4. Adapt to different types of problems
    """
    
    def __init__(self, active_model: str = "mistral"):
        self.active_model = active_model
        self.logger = logging.getLogger(__name__)
        
        # Initialize cognitive components
        self.planner = CognitivePlanner(active_model)
        self.reflection = SelfReflectionEngine(active_model)
        self.tree_of_thought = TreeOfThoughtEngine(active_model)
        self.commonsense = CommonsenseReasoner(active_model)
        
        # Performance tracking
        self.performance_history = []
        self.last_strategy_change = time.time()
        self.current_strategy = "balanced"  # balanced, creative, analytical
        
    async def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a user query with the most appropriate cognitive approach
        
        Args:
            query: The user's query
            context: Optional context information
            
        Returns:
            Response with reasoning details
        """
        # Determine the best cognitive approach for this query
        approach = await self._select_cognitive_approach(query, context)
        
        # Process the query using the selected approach
        if approach == "planning":
            result = await self._process_with_planning(query, context)
        elif approach == "tree_of_thought":
            result = await self._process_with_tot(query, context)
        elif approach == "commonsense":
            result = await self._process_with_commonsense(query, context)
        else:  # Default approach
            result = await self._process_with_planning(query, context)
            
        # Reflect on the result to improve future performance
        reflection = await self.reflection.reflect_on_response(query, result["text"], context)
        result["reflection"] = reflection
        
        # Track performance
        self._update_performance_tracking(approach, reflection["evaluation"]["overall_score"])
        
        return result
    
    async def _select_cognitive_approach(self, query: str, context: Optional[Dict]) -> str:
        """Select the most appropriate cognitive approach for this query"""
        # First, try to identify query type
        query_properties = {
            "length": len(query),
            "question_mark": "?" in query,
            "action_words": any(word in query.lower() for word in 
                              ["do", "make", "create", "build", "implement", "organize"]),
            "reasoning_words": any(word in query.lower() for word in 
                                 ["why", "how", "explain", "reason", "because", "therefore"]),
            "explore_words": any(word in query.lower() for word in
                               ["different", "options", "alternatives", "possibilities", "pros and cons"])
        }
        
        # Apply heuristics to select approach
        if query_properties["action_words"] and not query_properties["reasoning_words"]:
            # Task-oriented query - use planning approach
            return "planning"
        elif query_properties["reasoning_words"] or query_properties["explore_words"]:
            # Reasoning or exploration query - use tree of thought
            return "tree_of_thought"
        elif query_properties["length"] < 50 and not any([query_properties["action_words"], 
                                                       query_properties["reasoning_words"],
                                                       query_properties["explore_words"]]):
            # Short, simple query - likely needs commonsense
            return "commonsense"
        else:
            # Default to planning for most queries
            return "planning"
    
    async def _process_with_planning(self, query: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Process the query using the planning approach"""
        # Use planner to think about the problem
        thinking_result = await self.planner.think(query, context)
        
        # We would execute the plan with tools here, but for simplicity 
        # we'll just use the plan for our response
        
        # Format the response
        thoughts_summary = thinking_result["thoughts"]
        plan_steps = "\n".join([f"{i+1}. {step['description']}" 
                              for i, step in enumerate(thinking_result["steps"])])
        
        response_text = f"{plan_steps}"
        
        return {
            "text": response_text,
            "reasoning_approach": "planning",
            "thinking_process": thoughts_summary,
            "plan": plan_steps
        }
    
    async def _process_with_tot(self, query: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Process the query using the tree of thought approach"""
        # Use tree of thought to explore multiple reasoning paths
        reasoning_result = await self.tree_of_thought.solve(query, context)
        
        # Extract the solution
        solution = reasoning_result.get("solution", {})
        answer = solution.get("answer", "Could not determine an answer")
        thought_path = solution.get("thought_path", "")
        
        # Format the response
        response_text = answer
        
        return {
            "text": response_text,
            "reasoning_approach": "tree_of_thought",
            "thought_path": thought_path,
            "confidence": solution.get("confidence", 0)
        }
    
    async def _process_with_commonsense(self, query: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Process the query using commonsense reasoning"""
        # Infer relevant facts
        facts = await self.commonsense.infer_facts(query)
        
        # If it's a statement, check plausibility
        if not query.strip().endswith("?"):
            plausibility = await self.commonsense.check_plausibility(query)
            assessment = plausibility["assessment"]
            reasoning = plausibility["reasoning"]
            
            response_text = f"{assessment}. {reasoning}"
        else:
            # For questions, we would normally use a mix of techniques
            # But for simplicity, we'll just rely on facts
            facts_text = "\n".join([f"- {fact}" for fact in facts])
            response_text = f"Based on common sense:\n{facts_text}"
        
        return {
            "text": response_text,
            "reasoning_approach": "commonsense",
            "facts": facts
        }
    
    def _update_performance_tracking(self, approach: str, score: float) -> None:
        """Update performance tracking for adaptive strategy selection"""
        self.performance_history.append({
            "timestamp": time.time(),
            "approach": approach,
            "score": score
        })
        
        # Keep history manageable
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
            
        # Check if we should update our strategy
        current_time = time.time()
        if current_time - self.last_strategy_change > 3600:  # 1 hour
            self._update_cognitive_strategy()
            self.last_strategy_change = current_time
    
    def _update_cognitive_strategy(self) -> None:
        """Adaptively update cognitive strategy based on performance"""
        if not self.performance_history:
            return
            
        # Calculate average performance for each approach
        approach_scores = {}
        for entry in self.performance_history:
            approach = entry["approach"]
            score = entry["score"]
            
            if approach not in approach_scores:
                approach_scores[approach] = {"total": 0, "count": 0}
                
            approach_scores[approach]["total"] += score
            approach_scores[approach]["count"] += 1
            
        # Calculate averages
        for approach in approach_scores:
            if approach_scores[approach]["count"] > 0:
                approach_scores[approach]["average"] = (
                    approach_scores[approach]["total"] / approach_scores[approach]["count"]
                )
            else:
                approach_scores[approach]["average"] = 0
                
        # Find the best performing approach
        best_approach = max(approach_scores.items(), 
                           key=lambda x: x[1]["average"])
        
        # Update strategy based on best performance
        if best_approach[0] == "planning":
            self.current_strategy = "analytical"
        elif best_approach[0] == "tree_of_thought":
            self.current_strategy = "creative"
        else:
            self.current_strategy = "balanced"
            
        self.logger.info(f"Updated cognitive strategy to: {self.current_strategy}")
        
    async def get_cognitive_metrics(self) -> Dict[str, Any]:
        """Get metrics about cognitive performance and strategy"""
        if not self.performance_history:
            return {
                "strategy": self.current_strategy,
                "approach_performance": {},
                "reflection_metrics": self.reflection.get_reflection_metrics()
            }
            
        # Calculate metrics for each approach
        approach_metrics = {}
        for entry in self.performance_history:
            approach = entry["approach"]
            score = entry["score"]
            
            if approach not in approach_metrics:
                approach_metrics[approach] = {
                    "total": 0, 
                    "count": 0, 
                    "scores": []
                }
                
            approach_metrics[approach]["total"] += score
            approach_metrics[approach]["count"] += 1
            approach_metrics[approach]["scores"].append(score)
            
        # Calculate statistics
        for approach in approach_metrics:
            metrics = approach_metrics[approach]
            if metrics["count"] > 0:
                metrics["average"] = metrics["total"] / metrics["count"]
                metrics["min"] = min(metrics["scores"])
                metrics["max"] = max(metrics["scores"])
                
                if len(metrics["scores"]) > 1:
                    import numpy as np
                    metrics["std_dev"] = float(np.std(metrics["scores"]))
                else:
                    metrics["std_dev"] = 0.0
                    
            # Remove the raw scores to keep the output clean
            del metrics["scores"]
                
        return {
            "strategy": self.current_strategy,
            "approach_performance": approach_metrics,
            "reflection_metrics": self.reflection.get_reflection_metrics()
        }
    
    async def get_improvement_plan(self) -> Dict[str, Any]:
        """Get a plan for improving cognitive performance"""
        strategy = await self.reflection.get_improvement_strategy()
        
        # Add insights from performance tracking
        metrics = await self.get_cognitive_metrics()
        
        return {
            "current_strategy": self.current_strategy,
            "improvement_goals": strategy["goals"],
            "improvement_plan": strategy["strategy"],
            "performance_metrics": metrics
        } 