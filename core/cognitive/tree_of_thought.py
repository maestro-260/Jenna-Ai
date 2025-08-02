import ollama
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import re

logger = logging.getLogger(__name__)

class TreeOfThoughtEngine:
    """
    Implements Tree of Thought reasoning which enables:
    1. Exploring multiple reasoning paths simultaneously
    2. Evaluating the quality of different reasoning approaches
    3. Selecting the most promising path(s) to continue exploring
    4. Making better decisions by considering alternatives
    """
    
    def __init__(self, active_model: str = "mistral"):
        self.active_model = active_model
        self.logger = logging.getLogger(__name__)
        self.max_branches = 3  # Maximum reasoning branches to explore
        self.max_depth = 3     # Maximum reasoning depth to explore
        
    async def solve(self, problem: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Solve a complex problem using tree of thought reasoning
        
        Args:
            problem: The problem to solve
            context: Optional contextual information
            
        Returns:
            Dictionary with reasoning paths and final solution
        """
        # Generate initial thoughts (branches)
        initial_thoughts = await self._generate_initial_thoughts(problem)
        
        # Evaluate initial thoughts
        evaluated_thoughts = await self._evaluate_thoughts(initial_thoughts, problem)
        
        # Create reasoning tree
        reasoning_tree = {
            "problem": problem,
            "branches": []
        }
        
        # Explore promising branches
        for thought in evaluated_thoughts[:self.max_branches]:
            branch = await self._explore_branch(thought["thought"], problem, 1)
            reasoning_tree["branches"].append(branch)
            
        # Find the best solution
        solution = await self._select_best_solution(reasoning_tree)
        reasoning_tree["solution"] = solution
        
        return reasoning_tree
        
    async def _generate_initial_thoughts(self, problem: str) -> List[str]:
        """Generate initial potential approaches to the problem"""
        prompt = f"""
For this problem: "{problem}"

Generate {self.max_branches + 2} completely different approaches or initial thoughts for solving it.
Each approach should start from a different angle or use different methods.

Format your response as a JSON array of strings, with each string being a different approach.
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
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                thoughts = json.loads(json_match.group(0))
            else:
                # Extract lines that look like separate thoughts
                thoughts = [line.strip() for line in response_text.split('\n') 
                           if line.strip() and line.strip()[0].isdigit()]
                if not thoughts:
                    thoughts = ["Analyze the core components", 
                               "Use first principles thinking", 
                               "Consider analogous problems",
                               "Break the problem into smaller parts",
                               "Use a process of elimination"]
        except Exception as e:
            self.logger.error(f"Error parsing initial thoughts: {e}")
            thoughts = ["Analyze the core components", 
                       "Use first principles thinking", 
                       "Consider analogous problems",
                       "Break the problem into smaller parts", 
                       "Use a process of elimination"]
            
        return thoughts
    
    async def _evaluate_thoughts(self, thoughts: List[str], 
                                problem: str) -> List[Dict[str, Any]]:
        """Evaluate each thought for its promise in solving the problem"""
        evaluated_thoughts = []
        
        for thought in thoughts:
            prompt = f"""
For this problem: "{problem}"

Evaluate this approach: "{thought}"

Rate it on a scale from 1-10 where:
1 = Will definitely not lead to a solution
10 = Will definitely lead to a solution

Provide your rating and a brief explanation.
Format your response as a JSON object with "score" and "explanation" fields.
"""
            
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.active_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse the response
            response_text = response["message"]["content"]
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group(0))
                    score = int(evaluation.get("score", 5))
                    explanation = evaluation.get("explanation", "No explanation provided")
                else:
                    # Default score if parsing fails
                    score = 5
                    explanation = "Could not parse explanation"
            except Exception as e:
                self.logger.error(f"Error parsing thought evaluation: {e}")
                score = 5
                explanation = f"Error in evaluation: {str(e)}"
                
            evaluated_thoughts.append({
                "thought": thought,
                "score": score,
                "explanation": explanation
            })
            
        # Sort by score (descending)
        evaluated_thoughts.sort(key=lambda x: x["score"], reverse=True)
        return evaluated_thoughts
    
    async def _explore_branch(self, initial_thought: str, problem: str, 
                             depth: int) -> Dict[str, Any]:
        """Recursively explore a reasoning branch up to max_depth"""
        if depth >= self.max_depth:
            # Reached max depth, generate a conclusion
            conclusion = await self._generate_conclusion(initial_thought, problem)
            return {
                "thought": initial_thought,
                "conclusion": conclusion,
                "sub_thoughts": []
            }
            
        # Generate next level of thoughts
        sub_thoughts_prompt = f"""
Problem: "{problem}"
Current line of thinking: "{initial_thought}"

Generate 3 possible next steps or sub-thoughts that would follow from this thinking.
Each should explore a different direction while staying on the same general approach.

Format your response as a JSON array of strings.
"""
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": sub_thoughts_prompt}]
        )
        
        # Parse the response
        response_text = response["message"]["content"]
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                sub_thoughts = json.loads(json_match.group(0))
            else:
                sub_thoughts = []
        except Exception as e:
            self.logger.error(f"Error parsing sub-thoughts: {e}")
            sub_thoughts = []
            
        # Limit the number of sub-thoughts
        sub_thoughts = sub_thoughts[:self.max_branches]
        
        # Evaluate sub-thoughts
        evaluated_sub_thoughts = await self._evaluate_thoughts(sub_thoughts, problem)
        
        # Only explore the best sub-thought further (beam search)
        sub_branches = []
        for thought in evaluated_sub_thoughts[:1]:  # Only the best one
            sub_branch = await self._explore_branch(thought["thought"], problem, depth + 1)
            sub_branches.append(sub_branch)
            
        # Generate a conclusion for this branch
        conclusion = await self._generate_conclusion(initial_thought, problem)
            
        return {
            "thought": initial_thought,
            "conclusion": conclusion,
            "sub_thoughts": sub_branches
        }
    
    async def _generate_conclusion(self, reasoning_path: str, problem: str) -> str:
        """Generate a conclusion based on a reasoning path"""
        prompt = f"""
Problem: "{problem}"
Based on this line of reasoning: "{reasoning_path}"

What conclusion can you draw? Provide a specific, concise answer.
"""
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response["message"]["content"]
    
    async def _select_best_solution(self, reasoning_tree: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best solution from the reasoning tree"""
        branches = reasoning_tree["branches"]
        problem = reasoning_tree["problem"]
        
        # Collect all conclusions
        conclusions = []
        for branch in branches:
            if "conclusion" in branch:
                conclusions.append({
                    "thought_path": branch["thought"],
                    "conclusion": branch["conclusion"]
                })
            
            # Also check sub-branches
            if "sub_thoughts" in branch:
                for sub_branch in branch["sub_thoughts"]:
                    if "conclusion" in sub_branch:
                        thought_path = f"{branch['thought']} → {sub_branch['thought']}"
                        conclusions.append({
                            "thought_path": thought_path,
                            "conclusion": sub_branch["conclusion"]
                        })
        
        if not conclusions:
            return {
                "answer": "Could not generate a satisfactory answer.",
                "confidence": 0,
                "thought_path": "No complete reasoning paths found."
            }
            
        # Compare conclusions and select the best one
        conclusions_text = "\n\n".join([
            f"Path: {c['thought_path']}\nConclusion: {c['conclusion']}"
            for c in conclusions
        ])
        
        prompt = f"""
Problem: "{problem}"

Here are different possible conclusions:

{conclusions_text}

Which conclusion is best? Consider:
1. Logical correctness
2. Relevance to the original problem
3. Completeness of the answer
4. Practical applicability

Respond with a JSON object with these fields:
- best_conclusion_index (0-based index of the best conclusion)
- explanation (why this is the best)
- confidence (a number from 0-10)
- refined_answer (an improved version of the best conclusion)
"""
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the response
        response_text = response["message"]["content"]
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group(0))
                index = int(evaluation.get("best_conclusion_index", 0))
                explanation = evaluation.get("explanation", "No explanation provided")
                confidence = int(evaluation.get("confidence", 5))
                refined_answer = evaluation.get("refined_answer", conclusions[0]["conclusion"])
            else:
                # Default selection
                index = 0
                explanation = "Could not parse selection criteria"
                confidence = 5
                refined_answer = conclusions[0]["conclusion"]
        except Exception as e:
            self.logger.error(f"Error selecting best solution: {e}")
            index = 0
            explanation = f"Error in selection: {str(e)}"
            confidence = 5
            refined_answer = conclusions[0]["conclusion"]
            
        # Get the selected conclusion
        selected_conclusion = conclusions[min(index, len(conclusions)-1)]
        
        return {
            "answer": refined_answer,
            "confidence": confidence,
            "explanation": explanation,
            "thought_path": selected_conclusion["thought_path"]
        }

    def flatten_tree(self, reasoning_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert the tree structure to a flat list for easier analysis"""
        flat_paths = []
        
        def process_branch(branch, path_so_far):
            if not isinstance(branch, dict) or "thought" not in branch:
                return
                
            current_path = path_so_far + [branch["thought"]]
            
            # Add leaf nodes
            if not branch.get("sub_thoughts") or len(branch["sub_thoughts"]) == 0:
                flat_paths.append({
                    "path": current_path,
                    "conclusion": branch.get("conclusion", "No conclusion")
                })
            
            # Process sub-branches
            for sub_branch in branch.get("sub_thoughts", []):
                process_branch(sub_branch, current_path)
                
        # Process each top-level branch
        for branch in reasoning_tree.get("branches", []):
            process_branch(branch, [])
            
        return flat_paths