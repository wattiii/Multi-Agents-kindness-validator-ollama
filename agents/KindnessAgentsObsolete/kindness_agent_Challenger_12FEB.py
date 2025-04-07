import asyncio
import json
import time
from typing import Dict, Any
from loguru import logger
import edgedb
from config import save_ai_output_to_jsonl
from litellm import completion
import aiohttp

class OllamaLLM:
    """Custom Ollama LLM class for generating responses."""
    
    def __init__(self, model: str = "granite3.1-dense:2b", base_url: str = "http://localhost:11434", temperature: float = 0.2, max_tokens: int = 256):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate_response(self, prompt: str) -> str:
        """
        Call Ollama's API directly instead of using LiteLLM.
        Returns the raw text response.
        """
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}/api/generate"
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {error_text}")
                    
                    result = await response.json()
                    return result.get('response', '').strip()
                    
            except Exception as e:
                logger.error(f"Error calling Ollama API: {e}")
                return "0.0"  # Return default score on error


class FastKindnessEvaluator:
    """Evaluation module to assess AI responses."""
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    async def _score_metric(self, response: str, metric: str) -> bool:
        """Generic scoring method for any metric with debug logging."""
        prompt = f"""
        You are a thoughtful, helpful, intelligent, nice, and kind scoring assistant. 
        Rate the following text on how {metric} it is. 
        If the response meets the standard, return "True". 
        If the response does not meet the standard, return "False".
        Text: "{response}"
        """
        
        try:
            score_text = await self.llm.generate_response(prompt)
            print(f"\nRaw LLM Response for {metric} score: '{score_text}'")
            logger.debug(f"Raw score response: {score_text}")
            
            # Normalize and check if response is True or False
            score_text = score_text.strip().lower()
            if "True" in score_text or "true" in score_text or "yes" in score_text or score_text == "1":  # Handle variations for True
                score = 1
            elif "False" in score_text or "false" in score_text or "no" in score_text or score_text == "0":  # Handle variations for False
                score = 0
            else:
                score = 0  # Default to 0 if nothing matches
            
            return score
        
        except Exception as e:
            logger.error(f"Error generating {metric} score: {e}")
            return 0  # Return default score on error
            
    async def evaluate(self, prompt: str, response: str) -> Dict[str, Any]:
        """Evaluate the response and return the scores."""
        try:
            scores = await self.generate_scores(prompt, response)
            return scores
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return self.get_default_evaluation()

    async def generate_scores(self, prompt: str, response: str) -> Dict[str, float]:
        """Generate scores for the response based on prompt."""
        try:
            thoughtful_score = await self._score_metric(response, "thoughtful")
            helpful_score = await self._score_metric(response, "helpful")
            intelligent_score = await self._score_metric(response, "intelligent")
            nice_score = await self._score_metric(response, "nice")
            kind_score = await self._score_metric(response, "kind")
            
            overall_score = (thoughtful_score + helpful_score + intelligent_score + 
                           nice_score + kind_score) / 5

            return {
                "thoughtful_score": thoughtful_score,
                "helpful_score": helpful_score,
                "intelligent_score": intelligent_score,
                "nice_score": nice_score,
                "kind_score": kind_score,
                "overall_score": overall_score
            }
        except Exception as e:
            logger.error(f"Error in generate_scores: {e}")
            return self.get_default_evaluation()

    def get_default_evaluation(self) -> Dict[str, float]:
        """Return default evaluation when analysis fails."""
        return {
            "thoughtful_score": 0,
            "helpful_score": 1,
            "intelligent_score": 0,
            "nice_score": 1,
            "kind_score": 0,
            "overall_score": 0.4
        }
        
class FastKindnessAgent:
    """Agent to interact with the model and save results."""
    
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.llm = OllamaLLM()
        self.evaluator = FastKindnessEvaluator(llm=self.llm)
        self.client = edgedb.create_client()

    async def execute(self, prompt: str, ai_response: str) -> str:
        """Evaluate the response and save the results."""
        start_time = time.time()
        try:
            evaluation = await self.evaluator.evaluate(prompt, ai_response)
            await self.save_results(prompt, ai_response, evaluation)

            # Log execution time
            total_time = round((time.time() - start_time) * 1000)
            print(f"\n{'='*50}\nTotal execution time: {total_time}ms\n{'='*50}")
            
            return json.dumps(evaluation)
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            return json.dumps(await self.evaluator._get_default_evaluation())

    async def save_results(self, prompt: str, response: str, scores: Dict[str, float]):
        """Save results to JSONL and EdgeDB."""
        save_ai_output_to_jsonl(model_name=self.model_name, topic=prompt, content=response, jsonl_file_path="data/ai_outputs.jsonl", **scores)


# Example usage
async def main():
    """
    The main function creates a FastKindnessAgent instance, executes a task with a prompt and AI
    response, and prints the result.
    """
    agent = FastKindnessAgent()
    result = await agent.execute(prompt="How do I bake cookies?", ai_response="Here's a detailed recipe...")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
