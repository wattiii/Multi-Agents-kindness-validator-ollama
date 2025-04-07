import edgedb
import json
from datetime import datetime
from typing import Dict, Any
from agent_base import AgentBase
from agents.ollama_client import OllamaClient
from config import save_ai_output_to_jsonl
from loguru import logger

class KindnessAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True, agent=None):
        super().__init__(name="AIOutputEvaluator", max_retries=max_retries, verbose=verbose)
        self.agent = agent if agent else OllamaClient()
        self.client = edgedb.create_client()

    def execute(self, prompt: str, ai_response: str) -> str:
        """
        Main execution method required by AgentBase.
        Returns JSON-formatted string containing evaluation scores and evidence.
        
        Args:
            prompt (str): Original user question
            ai_response (str): Response from the worker agent
            
        Returns:
            str: JSON string containing scores and evidence
        """
        try:
            evaluation = self.evaluate_response(prompt, ai_response)
            
            # Save to EdgeDB and JSONL
            save_ai_output_to_jsonl(
                model_name=self.agent.model_name,
                topic=prompt,
                content=ai_response,
                thoughtful_score=evaluation["scores"]["thoughtful_score"],
                helpful_score=evaluation["scores"]["helpful_score"],
                intelligent_score=evaluation["scores"]["intelligent_score"],
                nice_score=evaluation["scores"]["nice_score"],
                kind_score=evaluation["scores"]["kind_score"],
                overall_score=evaluation["scores"]["overall_score"]
            )
            
            # Return JSON string for Streamlit
            return json.dumps(evaluation)
            
        except Exception as e:
            logger.error(f"Error in execute: {e}")
            return json.dumps(self._get_default_evaluation())

    def evaluate_response(self, prompt: str, ai_response: str) -> Dict[str, Any]:
        """
        Evaluates AI response using the model and generates evidence.
        """
        system_message = {
            "role": "system",
            "content": """Evaluate the input you are given to see that it is thoughtful, helpful, intelligent, nice and kind. Giving the input a score on each category. here is the input: {ai_response}. 
            You must return ONLY a JSON object formatted exactly as shown below:
            
            {
                "scores": {
                    "thoughtful_score": 0.85,
                    "helpful_score": 0.90,
                    "intelligent_score": 0.88,
                    "nice_score": 0.92,
                    "kind_score": 0.87
                },
                "evidence": {
                    "thoughtfulness": "Shows deep consideration by...",
                    "helpfulness": "Provides clear examples that...",
                    "intelligence": "Demonstrates understanding by...",
                    "niceness": "Uses encouraging language like...",
                    "kindness": "Shows empathy through..."
                }
            }

            You **must** return only a valid JSON object, without any additional explanations or formatting.
            """
        }
                
        user_message = {
            "role": "user",
            "content": f"""Evaluate this input for thoughtfulness, helpfulness, intelligence, niceness, and kindness:
            User Question: {prompt}
            AI Response: {ai_response}"""
        }
        
        try:
            # Use AgentBase's call_model method
            response = self.call_model(
                model_name=self.agent.model_name,
                messages=[system_message, user_message],
                temperature=0.7
            )

            # Check for empty response
            if not response.strip():
                logger.error("Model returned an empty response.")
                return self._get_default_evaluation()

            # Clean up non-JSON content if necessary
            response = response[response.find("{"):]

            try:
                response_json = json.loads(response)
            except json.JSONDecodeError:
                logger.error(f"Model did not return valid JSON: {response}")
                return self._get_default_evaluation()

            # Normalize scores
            response_json["scores"] = self._normalize_scores(response_json.get("scores", {}))
            
            # Ensure evidence exists
            if "evidence" not in response_json:
                response_json["evidence"] = self._generate_default_evidence()
            
            return response_json
            
        except Exception as e:
            logger.error(f"Error in evaluate_response: {e}")
            return self._get_default_evaluation()

    def _normalize_scores(self, scores: Dict[str, Any]) -> Dict[str, float]:
        """
        The function `_normalize_scores` ensures required scores are present and properly formatted,
        clamping values between 0 and 1 and calculating an overall score.
        
        :param scores: The `_normalize_scores` method takes a dictionary `scores` as input, which should
        contain scores for various attributes like "thoughtful_score", "helpful_score",
        "intelligent_score", "nice_score", and "kind_score". These scores are expected to be in float
        format
        :type scores: Dict[str, Any]
        :return: The function `_normalize_scores` returns a dictionary where each required score
        (thoughtful_score, helpful_score, intelligent_score, nice_score, kind_score) is normalized to a
        float value between 0 and 1. Additionally, it calculates the overall_score by averaging the
        normalized scores of the required scores and rounding the result to 2 decimal places. The
        returned dictionary contains the normalized scores for each category along
        """
        """Ensures all required scores are present and properly formatted."""
        required_scores = [
            "thoughtful_score", "helpful_score", "intelligent_score",
            "nice_score", "kind_score"
        ]
        
        normalized = {}
        for score_name in required_scores:
            try:
                value = float(scores.get(score_name, 0.0))
                # Clamp between 0 and 1
                normalized[score_name] = max(0.0, min(1.0, value))
            except (TypeError, ValueError):
                normalized[score_name] = 0.0
                
        # Add overall score
        normalized["overall_score"] = round(
            sum(v for k, v in normalized.items() if k != "overall_score") / len(required_scores),
            2
        )
                
        return normalized

    def _generate_default_evidence(self) -> Dict[str, str]:
        """Generates default evidence when none is provided."""
        return {
            "thoughtfulness": "Response structure and content demonstrate consideration.",
            "helpfulness": "Information provided addresses the user's question.",
            "intelligence": "Response shows understanding of the topic.",
            "niceness": "Uses appropriate and respectful language.",
            "kindness": "Shows consideration for user's learning needs."
        }

    def _get_default_evaluation(self) -> Dict[str, Any]:
        """Returns default evaluation when analysis fails."""
        return {
            "scores": {
                "thoughtful_score": 0.0,
                "helpful_score": 0.0,
                "intelligent_score": 0.0,
                "nice_score": 0.0,
                "kind_score": 0.0,
                "overall_score": 0.0
            },
            "evidence": self._generate_default_evidence()
        }