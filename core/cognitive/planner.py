import asyncio
import logging
from typing import List, Dict, Any, Optional
import re
import ollama
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class CognitivePlanner:
    """
    Implements ReAct (Reasoning and Acting) for more human-like thinking and task execution.
    This enables the AI to:
    1. Break down complex tasks into steps
    2. Reason about each step before executing
    3. Adapt to feedback during execution
    4. Plan task sequences dynamically
    """
    
    def __init__(self, active_model: str = "mistral"):
        self.active_model = active_model
        self.logger = logging.getLogger(__name__)
        self.thinking_depth = 3  # Number of thinking steps before action
        self.react_template = """
You are JENNA, an intelligent AI assistant. You need to think step-by-step to solve this problem:

{problem}

Follow this thinking process:
1. Analyze the problem thoroughly
2. Consider what information you need
3. Break the problem into smaller steps
4. Decide what actions to take in what order
5. Execute each step carefully

First, break down your thinking process. Then form a plan. Then execute.

Thought: {thoughts}

Plan: 
"""

    async def think(self, problem: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform a structured thinking process on a problem
        
        Args:
            problem: The problem or task to solve
            context: Optional additional context
            
        Returns:
            Dictionary with thinking steps, plan, and execution details
        """
        # Generate initial thoughts through chain-of-thought
        thoughts = await self._generate_thoughts(problem)
        
        # Create a structured plan from thoughts
        plan = await self._create_plan(problem, thoughts)
        
        # Break down plan into executable steps
        steps = self._extract_steps(plan)
        
        return {
            "thoughts": thoughts,
            "plan": plan,
            "steps": steps,
            "problem": problem,
            "completed": False
        }
    
    async def _generate_thoughts(self, problem: str) -> str:
        """Generate structured thinking about the problem"""
        # Use a recursive thinking process to go deeper
        thoughts = []
        current_thought = "I need to understand what's being asked."
        
        for i in range(self.thinking_depth):
            prompt = f"Problem: {problem}\nPrevious thinking: {current_thought}\nNext thought:"
            
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.active_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            current_thought = response["message"]["content"]
            thoughts.append(current_thought)
        
        return "\n".join(thoughts)
    
    async def _create_plan(self, problem: str, thoughts: str) -> str:
        """Create a structured plan based on thoughts"""
        prompt_template = PromptTemplate.from_template(self.react_template)
        prompt = prompt_template.format(problem=problem, thoughts=thoughts)
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response["message"]["content"]
    
    def _extract_steps(self, plan: str) -> List[Dict[str, Any]]:
        """Extract executable steps from the plan"""
        # Look for numbered steps in the plan
        step_pattern = r"(?:^|\n)(\d+)[\.:\)]\s*(.+?)(?=\n\d+[\.:\)]|\Z)"
        matches = re.finditer(step_pattern, plan, re.MULTILINE | re.DOTALL)
        
        steps = []
        for match in matches:
            step_num = match.group(1)
            step_text = match.group(2).strip()
            steps.append({
                "id": int(step_num),
                "description": step_text,
                "completed": False,
                "result": None
            })
        
        # If we couldn't find structured steps, create a single step
        if not steps:
            steps = [{
                "id": 1,
                "description": plan.strip(),
                "completed": False,
                "result": None
            }]
            
        return steps
    
    async def execute_step(self, step: Dict[str, Any], tools: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step of the plan using available tools
        
        Args:
            step: The step to execute
            tools: Dictionary of available tools
            
        Returns:
            Updated step with results
        """
        step_description = step["description"]
        
        # Determine which tool to use for this step
        tool_prompt = f"""
Based on this step: "{step_description}"
Which of these tools would be best to use: {', '.join(tools.keys())}
Respond with JUST the tool name, nothing else.
"""
        
        tool_response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": tool_prompt}]
        )
        
        tool_name = tool_response["message"]["content"].strip().lower()
        
        # Find the matching tool
        selected_tool = None
        for name, tool in tools.items():
            if name.lower() in tool_name:
                selected_tool = tool
                break
        
        # Use a default approach if no tool matches
        if not selected_tool:
            self.logger.warning(f"No tool found for step: {step_description}")
            step["result"] = "Could not find appropriate tool for this step"
            return step
            
        # Generate parameters for the tool
        param_prompt = f"""
For this step: "{step_description}"
I need to use the {tool_name} tool.
What parameters should I pass to this tool? Respond in JSON format with parameter names and values.
"""
        
        param_response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": param_prompt}]
        )
        
        # Extract parameters from response
        param_text = param_response["message"]["content"]
        try:
            # Try to find JSON in the response
            import json
            json_match = re.search(r'```json\n(.*?)\n```', param_text, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group(1))
            else:
                # Try to find a JSON object directly
                json_match = re.search(r'\{.*\}', param_text, re.DOTALL)
                if json_match:
                    params = json.loads(json_match.group(0))
                else:
                    params = {}
        except Exception as e:
            self.logger.error(f"Error parsing parameters: {e}")
            params = {}
            
        # Execute the tool with parameters
        try:
            result = await selected_tool(**params)
            step["result"] = result
            step["completed"] = True
        except Exception as e:
            self.logger.error(f"Error executing tool: {e}")
            step["result"] = f"Error: {str(e)}"
            
        return step
    
    async def execute_plan(self, cognitive_result: Dict[str, Any], tools: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all steps in the plan
        
        Args:
            cognitive_result: The result from the think method
            tools: Dictionary of available tools
            
        Returns:
            Updated cognitive result with execution results
        """
        steps = cognitive_result["steps"]
        
        for i, step in enumerate(steps):
            updated_step = await self.execute_step(step, tools)
            steps[i] = updated_step
            
            # Check if we need to adjust the plan based on this step's result
            if i < len(steps) - 1 and not updated_step["completed"]:
                # We hit a problem, we might need to replan
                remaining_steps = steps[i+1:]
                new_problem = f"Previous step failed: {updated_step['description']} with result: {updated_step['result']}. Need to adjust plan for remaining steps: {remaining_steps}"
                
                # Replan the remaining steps
                new_thoughts = await self._generate_thoughts(new_problem)
                new_plan = await self._create_plan(new_problem, new_thoughts) 
                new_steps = self._extract_steps(new_plan)
                
                # Replace remaining steps with new plan
                cognitive_result["steps"] = steps[:i+1] + new_steps
                return await self.execute_plan(cognitive_result, tools)
        
        cognitive_result["completed"] = all(step["completed"] for step in steps)
        return cognitive_result
    
    async def reflect(self, cognitive_result: Dict[str, Any]) -> str:
        """
        Reflect on the execution process and results
        
        Args:
            cognitive_result: The result from executing a plan
            
        Returns:
            Reflection on the process and results
        """
        steps = cognitive_result["steps"]
        successful_steps = [step for step in steps if step["completed"]]
        failed_steps = [step for step in steps if not step["completed"]]
        
        reflection_prompt = f"""
I was given this problem: {cognitive_result['problem']}

My thinking process was:
{cognitive_result['thoughts']}

My plan was:
{cognitive_result['plan']}

Results of execution:
Successful steps: {len(successful_steps)}/{len(steps)}
Failed steps: {len(failed_steps)}/{len(steps)}

Step details:
{[f"Step {step['id']}: {step['description']} - {'Completed' if step['completed'] else 'Failed'}" for step in steps]}

Based on these results, what have I learned? How could I improve my approach next time?
"""
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": reflection_prompt}]
        )
        
        return response["message"]["content"] 