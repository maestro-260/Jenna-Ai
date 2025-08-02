import asyncio
import logging
from typing import Dict, List, Any, Optional
import json
import ollama
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

class CommonsenseReasoner:
    """
    Provides commonsense reasoning capabilities to help the AI understand everyday situations.
    This enables the AI to:
    1. Draw on world knowledge for practical reasoning
    2. Infer unstated facts and relationships
    3. Identify cause-effect relationships
    4. Make reasonable predictions about outcomes
    """
    
    def __init__(self, active_model: str = "mistral"):
        self.active_model = active_model
        self.logger = logging.getLogger(__name__)
        self._generative_model = None
        self._tokenizer = None
        
    async def _load_knowledge_model(self):
        """Lazy-load specialized commonsense reasoning model"""
        if self._generative_model is None:
            try:
                # Load specialized commonsense model
                self._tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m-bimodal")
                self._generative_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-220m-bimodal")
                self.logger.info("Loaded specialized commonsense model")
            except Exception as e:
                self.logger.error(f"Failed to load specialized model: {e}")
                # Will fallback to ollama for reasoning
                
    async def infer_facts(self, context: str) -> List[str]:
        """
        Infer unstated facts from the given context
        
        Args:
            context: The context to reason about
            
        Returns:
            List of inferred facts
        """
        prompt = f"""
Given this context: "{context}"

Infer 5 unstated facts that are likely to be true.
Focus on commonsense knowledge that would be obvious to humans.
Format your response as a JSON array of strings.
"""
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the response
        response_text = response["message"]["content"]
        try:
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                facts = json.loads(json_match.group(0))
            else:
                # Extract lines that look like facts
                facts = [line.strip() for line in response_text.split('\n') 
                       if line.strip() and line.strip()[0].isdigit()]
                if not facts:
                    facts = ["No facts could be inferred"]
        except Exception as e:
            self.logger.error(f"Error parsing inferred facts: {e}")
            facts = ["Error inferring facts"]
            
        return facts
    
    async def predict_outcomes(self, scenario: str, actions: List[str]) -> Dict[str, List[str]]:
        """
        Predict potential outcomes for different actions in a scenario
        
        Args:
            scenario: The scenario to reason about
            actions: List of possible actions
            
        Returns:
            Dictionary mapping actions to potential outcomes
        """
        results = {}
        
        for action in actions:
            prompt = f"""
In this scenario: "{scenario}"

If someone takes this action: "{action}"

What are the 3 most likely outcomes? Consider:
1. Immediate effects
2. Possible complications
3. Long-term consequences

Format your response as a JSON array of strings.
"""
            
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.active_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse the response
            response_text = response["message"]["content"]
            try:
                import re
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    outcomes = json.loads(json_match.group(0))
                else:
                    # Extract lines that look like outcomes
                    outcomes = [line.strip() for line in response_text.split('\n') 
                              if line.strip() and line.strip()[0].isdigit()]
                    if not outcomes:
                        outcomes = ["Outcome unclear"]
            except Exception as e:
                self.logger.error(f"Error parsing outcomes: {e}")
                outcomes = ["Error predicting outcomes"]
                
            results[action] = outcomes
            
        return results
    
    async def identify_cause_effect(self, event: str) -> Dict[str, List[str]]:
        """
        Identify potential causes and effects for an event
        
        Args:
            event: The event to analyze
            
        Returns:
            Dictionary with potential causes and effects
        """
        # Load specialized model if needed
        await self._load_knowledge_model()
        
        # Use specialized model if available, otherwise fall back to ollama
        if self._generative_model is not None:
            # Generate causes with specialized model
            causes = await self._generate_with_specialized_model(
                f"What are the most likely causes of this event: {event}"
            )
            
            # Generate effects with specialized model
            effects = await self._generate_with_specialized_model(
                f"What are the most likely effects of this event: {event}"
            )
        else:
            # Fallback to ollama
            causes_prompt = f"""
What are the 3 most likely causes for this event: "{event}"
Format your response as a JSON array of strings.
"""
            
            effects_prompt = f"""
What are the 3 most likely effects of this event: "{event}"
Format your response as a JSON array of strings.
"""
            
            causes_response = await asyncio.to_thread(
                ollama.chat,
                model=self.active_model,
                messages=[{"role": "user", "content": causes_prompt}]
            )
            
            effects_response = await asyncio.to_thread(
                ollama.chat,
                model=self.active_model,
                messages=[{"role": "user", "content": effects_prompt}]
            )
            
            # Parse responses
            try:
                import re
                causes_json = re.search(r'\[.*\]', causes_response["message"]["content"], re.DOTALL)
                effects_json = re.search(r'\[.*\]', effects_response["message"]["content"], re.DOTALL)
                
                if causes_json:
                    causes = json.loads(causes_json.group(0))
                else:
                    causes = ["Cause unclear"]
                    
                if effects_json:
                    effects = json.loads(effects_json.group(0))
                else:
                    effects = ["Effect unclear"]
            except Exception as e:
                self.logger.error(f"Error parsing cause-effect: {e}")
                causes = ["Error identifying causes"]
                effects = ["Error identifying effects"]
        
        return {
            "causes": causes,
            "effects": effects
        }
    
    async def _generate_with_specialized_model(self, prompt: str) -> List[str]:
        """Generate response using the specialized model"""
        try:
            # Encode and generate
            inputs = self._tokenizer(prompt, return_tensors="pt")
            outputs = self._generative_model.generate(
                inputs["input_ids"], 
                max_length=100,
                num_return_sequences=3
            )
            
            # Decode outputs
            decoded_outputs = [
                self._tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            # Clean and return
            results = [output.strip() for output in decoded_outputs if output.strip()]
            return results or ["No result generated"]
            
        except Exception as e:
            self.logger.error(f"Error generating with specialized model: {e}")
            return ["Error in specialized model generation"]
    
    async def check_plausibility(self, statement: str) -> Dict[str, Any]:
        """
        Check the plausibility of a statement based on commonsense knowledge
        
        Args:
            statement: The statement to evaluate
            
        Returns:
            Dictionary with plausibility score and reasoning
        """
        prompt = f"""
Evaluate the plausibility of this statement based on commonsense knowledge:

"{statement}"

Rate it on a scale from 1 to 10 where:
1 = Completely implausible, violates fundamental facts about the world
10 = Completely plausible, aligns with common knowledge

Provide your rating and reasoning.
Format your response as a JSON object with "score" and "reasoning" fields.
"""
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the response
        response_text = response["message"]["content"]
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group(0))
                score = float(evaluation.get("score", 5))
                reasoning = evaluation.get("reasoning", "No reasoning provided")
            else:
                # Default if parsing fails
                score = 5
                reasoning = "Could not parse reasoning"
        except Exception as e:
            self.logger.error(f"Error parsing plausibility check: {e}")
            score = 5
            reasoning = f"Error in evaluation: {str(e)}"
            
        # Determine categorical assessment
        assessment = "Unknown"
        if score >= 7:
            assessment = "Plausible"
        elif score <= 3:
            assessment = "Implausible"
        else:
            assessment = "Uncertain"
            
        return {
            "score": score,
            "reasoning": reasoning,
            "assessment": assessment
        }
    
    async def relate_concepts(self, concept1: str, concept2: str) -> Dict[str, Any]:
        """
        Find relationships between two concepts using commonsense knowledge
        
        Args:
            concept1: First concept
            concept2: Second concept
            
        Returns:
            Dictionary with relationships and explanations
        """
        prompt = f"""
Identify all meaningful relationships between these two concepts:

Concept 1: "{concept1}"
Concept 2: "{concept2}"

Consider:
1. Categorical relationships (is-a, part-of, etc.)
2. Functional relationships (used-for, enables, etc.)
3. Temporal/causal relationships (leads-to, precedes, etc.)
4. Similarity or contrast
5. Common contexts where they co-occur

Format your response as a JSON object with these fields:
- primary_relationship (the strongest relationship)
- all_relationships (array of all identified relationships)
- explanation (detailed explanation of how these concepts relate)
"""
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.active_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the response
        response_text = response["message"]["content"]
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                primary = data.get("primary_relationship", "Unknown relationship")
                all_rels = data.get("all_relationships", ["Unknown relationship"])
                explanation = data.get("explanation", "No explanation provided")
            else:
                # Default if parsing fails
                primary = "Unknown relationship"
                all_rels = ["Unknown relationship"]
                explanation = response_text  # Use the whole response as explanation
        except Exception as e:
            self.logger.error(f"Error parsing concept relationships: {e}")
            primary = "Error finding relationships"
            all_rels = ["Error finding relationships"]
            explanation = f"Error in analysis: {str(e)}"
            
        return {
            "primary_relationship": primary,
            "all_relationships": all_rels,
            "explanation": explanation
        } 