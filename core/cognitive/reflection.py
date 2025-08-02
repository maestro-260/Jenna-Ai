import ollama
import asyncio
import logging
from typing import Dict, List, Any, Optional
import json
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class SelfReflectionEngine:
    """
    A metacognitive system that enables self-reflection, critique, and improvement.
    This enables the AI to:
    1. Evaluate the quality of its own outputs
    2. Identify strengths and weaknesses
    3. Maintain improvement goals
    4. Track learning progress over time
    """
    
    def __init__(self, active_model: str = "mistral"):
        self.active_model = active_model
        self.logger = logging.getLogger(__name__)
        self.reflection_history = []
        self.improvement_goals = []
        
    async def reflect_on_response(self, user_input: str, ai_response: str, 
                                 context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Reflect on a response to identify strengths and weaknesses
        
        Args:
            user_input: The original user input
            ai_response: The AI's response
            context: Optional context information
            
        Returns:
            Dictionary with reflection results
        """
        # Generate evaluation criteria based on context
        criteria = await self._generate_criteria(user_input, context)
        
        # Evaluate response against criteria
        evaluation = await self._evaluate_response(user_input, ai_response, criteria)
        
        # Generate improvement suggestions
        improvements = await self._generate_improvements(evaluation)
        
        # Record this reflection
        timestamp = datetime.now().isoformat()
        reflection = {
            "timestamp": timestamp,
            "user_input": user_input,
            "ai_response": ai_response,
            "criteria": criteria,
            "evaluation": evaluation,
            "improvements": improvements
        }
        self.reflection_history.append(reflection)
        
        # Update improvement goals based on this reflection
        await self._update_improvement_goals(reflection)
        
        return reflection
    
    async def _generate_criteria(self, user_input: str, 
                                context: Optional[Dict] = None) -> List[str]:
        """Generate relevant evaluation criteria based on the input"""
        prompt = f"""
Given this user input: "{user_input}"

Generate 5 specific criteria that would be most important for evaluating a high-quality response.
Each criterion should be a single phrase or short sentence.
Format your response as a JSON array of strings.
"""
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Try to extract JSON array
        response_text = response["message"]["content"]
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                criteria = json.loads(json_match.group(0))
            else:
                # Fallback to default criteria
                criteria = [
                    "Relevance to query",
                    "Accuracy of information",
                    "Clarity of explanation",
                    "Completeness of answer",
                    "Appropriate tone"
                ]
        except Exception as e:
            self.logger.error(f"Error parsing criteria: {e}")
            criteria = [
                "Relevance to query",
                "Accuracy of information",
                "Clarity of explanation",
                "Completeness of answer",
                "Appropriate tone"
            ]
            
        return criteria
    
    async def _evaluate_response(self, user_input: str, ai_response: str, 
                                criteria: List[str]) -> Dict[str, Any]:
        """Evaluate the response against each criterion"""
        
        scores = {}
        explanations = {}
        
        for criterion in criteria:
            prompt = f"""
Evaluate this AI response against this specific criterion: "{criterion}"

User input: "{user_input}"
AI response: "{ai_response}"

Rate on a scale from 1-10 where:
1 = Completely fails to meet the criterion
10 = Perfectly meets the criterion

Provide your rating and a brief explanation.
Format your response as a JSON object with "score" and "explanation" fields.
"""
            
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.active_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse the response to extract score and explanation
            response_text = response["message"]["content"]
            try:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group(0))
                    scores[criterion] = int(evaluation.get("score", 5))
                    explanations[criterion] = evaluation.get("explanation", "No explanation provided.")
                else:
                    # Default if no JSON found
                    scores[criterion] = 5
                    explanations[criterion] = "Could not parse explanation."
            except Exception as e:
                self.logger.error(f"Error parsing evaluation: {e}")
                scores[criterion] = 5
                explanations[criterion] = "Error in evaluation."
        
        # Calculate overall score
        overall_score = np.mean(list(scores.values()))
        
        return {
            "scores": scores,
            "explanations": explanations,
            "overall_score": float(overall_score)
        }
    
    async def _generate_improvements(self, evaluation: Dict[str, Any]) -> List[str]:
        """Generate specific improvement suggestions based on evaluation"""
        # Find the lowest scoring criteria
        scores = evaluation["scores"]
        explanations = evaluation["explanations"]
        
        # Combine scores and explanations
        criteria_feedback = []
        for criterion, score in scores.items():
            explanation = explanations.get(criterion, "")
            criteria_feedback.append(f"{criterion}: {score}/10 - {explanation}")
        
        prompt = f"""
Based on this evaluation of an AI response:

{criteria_feedback}

Generate 3 specific, actionable improvements that would address the weakest areas.
Format your response as a JSON array of strings.
"""
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Try to extract JSON array
        response_text = response["message"]["content"]
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                improvements = json.loads(json_match.group(0))
            else:
                # Extract lines that look like improvements
                improvements = [line.strip() for line in response_text.split('\n') 
                               if line.strip() and line.strip()[0].isdigit()]
                if not improvements:
                    improvements = ["Improve clarity", "Add more specific details", "Maintain focus on the query"]
        except Exception as e:
            self.logger.error(f"Error parsing improvements: {e}")
            improvements = ["Improve clarity", "Add more specific details", "Maintain focus on the query"]
            
        return improvements
    
    async def _update_improvement_goals(self, reflection: Dict[str, Any]) -> None:
        """Update improvement goals based on reflection insights"""
        # Get current improvement goals
        current_goals = self.improvement_goals
        
        # Extract new improvements
        new_improvements = reflection["improvements"]
        evaluation = reflection["evaluation"]
        
        # Find the weakest criteria (lowest scores)
        scores = evaluation["scores"]
        weak_areas = [criterion for criterion, score in scores.items() if score < 6]
        
        # Combine with improvements to create potential new goals
        potential_goals = []
        for area in weak_areas:
            for improvement in new_improvements:
                potential_goals.append(f"Improve {area}: {improvement}")
        
        # Keep only the top 5 goals (or fewer if not enough)
        self.improvement_goals = (current_goals + potential_goals)[:5]
    
    async def get_improvement_strategy(self) -> Dict[str, Any]:
        """Generate a strategy to achieve current improvement goals"""
        if not self.improvement_goals:
            return {
                "goals": [],
                "strategy": "No improvement goals set yet."
            }
        
        goals_text = "\n".join([f"- {goal}" for goal in self.improvement_goals])
        
        prompt = f"""
I want to improve on these specific goals:

{goals_text}

Create a concrete strategy to achieve these improvements. Include:
1. Specific techniques for each goal
2. How to measure progress
3. A learning plan with steps

Format your response as a detailed plan.
"""
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        strategy = response["message"]["content"]
        
        return {
            "goals": self.improvement_goals,
            "strategy": strategy
        }
    
    def get_reflection_metrics(self) -> Dict[str, Any]:
        """Get metrics about reflection history and improvement over time"""
        if not self.reflection_history:
            return {
                "total_reflections": 0,
                "average_score": 0.0,
                "trend": "No data available"
            }
        
        # Calculate metrics
        total_reflections = len(self.reflection_history)
        
        # Get scores and timestamps
        scores = []
        timestamps = []
        for reflection in self.reflection_history:
            if "evaluation" in reflection and "overall_score" in reflection["evaluation"]:
                scores.append(reflection["evaluation"]["overall_score"])
                timestamps.append(reflection["timestamp"])
        
        # Calculate trend if we have enough data
        trend = "Stable"
        if len(scores) >= 5:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg * 1.05:
                trend = "Improving"
            elif second_avg < first_avg * 0.95:
                trend = "Declining"
            else:
                trend = "Stable"
        
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "total_reflections": total_reflections,
            "average_score": average_score,
            "trend": trend,
            "current_goals": self.improvement_goals
        } 