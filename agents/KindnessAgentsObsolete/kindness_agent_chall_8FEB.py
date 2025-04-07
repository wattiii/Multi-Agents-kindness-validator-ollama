import edgedb
import json
import multiprocessing
from datetime import datetime
from typing import Dict, Any, List
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
            
            return json.dumps(evaluation)
        except Exception as e:
            logger.error(f"Error in execute: {e}")
            return json.dumps(self._get_default_evaluation())

    def evaluate_response(self, prompt: str, ai_response: str) -> Dict[str, Any]:
        system_message = {
            "role": "system",
            "content": f"""Evaluate the input for thoughtfulness, helpfulness, intelligence, niceness, and kindness.
            User Question: {prompt}
            AI Response: {ai_response}
            You must return ONLY a JSON object formatted as follows:
            {{
                "scores": {{
                    "thoughtful_score": 0.85,
                    "helpful_score": 0.90,
                    "intelligent_score": 0.88,
                    "nice_score": 0.92,
                    "kind_score": 0.87
                }},
                "evidence": {{
                    "thoughtfulness": "Shows deep consideration by...",
                    "helpfulness": "Provides clear examples that...",
                    "intelligence": "Demonstrates understanding by...",
                    "niceness": "Uses encouraging language like...",
                    "kindness": "Shows empathy through..."
                }}
            }}
            """
        }
        
        user_message = {
            "role": "user",
            "content": f"Evaluate this response: {ai_response}"
        }
        
        try:
            response = self.call_model(
                model_name=self.agent.model_name,
                messages=[system_message, user_message],
                temperature=0.7
            )
            if not response.strip():
                logger.error("Model returned an empty response.")
                return self._get_default_evaluation()
            response = response[response.find("{"):]  # Ensure JSON formatting
            try:
                response_json = json.loads(response)
            except json.JSONDecodeError:
                logger.error(f"Model did not return valid JSON: {response}")
                return self._get_default_evaluation()
            response_json["scores"] = self._normalize_scores(response_json.get("scores", {}))
            if "evidence" not in response_json:
                response_json["evidence"] = self._generate_default_evidence()
            return response_json
        except Exception as e:
            logger.error(f"Error in evaluate_response: {e}")
            return self._get_default_evaluation()

    def evaluate_responses_parallel(self, responses: List[Dict[str, str]], num_workers=4) -> List[Dict[str, Any]]:
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.starmap(self.evaluate_response, [(r["prompt"], r["ai_response"]) for r in responses])
        return results

    def _normalize_scores(self, scores: Dict[str, Any]) -> Dict[str, float]:
        required_scores = [
            "thoughtful_score", "helpful_score", "intelligent_score",
            "nice_score", "kind_score"
        ]
        normalized = {}
        for score_name in required_scores:
            try:
                value = float(scores.get(score_name, 0.0))
                normalized[score_name] = max(0.0, min(1.0, value))
            except (TypeError, ValueError):
                normalized[score_name] = 0.0
        normalized["overall_score"] = round(
            sum(v for k, v in normalized.items() if k != "overall_score") / len(required_scores),
            2
        )
        return normalized

    def _generate_default_evidence(self) -> Dict[str, str]:
        return {
            "thoughtfulness": "Response structure and content demonstrate consideration.",
            "helpfulness": "Information provided addresses the user's question.",
            "intelligence": "Response shows understanding of the topic.",
            "niceness": "Uses appropriate and respectful language.",
            "kindness": "Shows consideration for user's learning needs."
        }

    def _get_default_evaluation(self) -> Dict[str, Any]:
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
