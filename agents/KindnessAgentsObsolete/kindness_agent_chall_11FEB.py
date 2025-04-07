import asyncio
import json
import time
from typing import Dict, Any
from loguru import logger
import edgedb
from config import save_ai_output_to_jsonl
from litellm import completion
import aiohttp
import logging
from worker import WorkerAgent

logger = logging.getLogger(__name__)

class OllamaLLM:
    """Custom Ollama LLM class for generating responses."""

    def __init__(self, model: str = "granite3.1-dense:2b", base_url: str = "http://localhost:11434", temperature: float = 0.2, max_tokens: int = 256):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._session = None
        self._session_owner = False  # Track whether we created the session

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        self._session_owner = True  # Mark that we own this session
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session and self._session_owner:
            await self._session.close()
            self._session = None

    async def generate_response(self, prompt: str) -> str:
        """
        Call Ollama's API directly instead of using LiteLLM.
        Ensures session is closed if it's created inside this function.
        """
        close_after =True  # Track if we should close the session

        if not self._session:
            self._session = aiohttp.ClientSession()
            close_after = True  # Mark that we should close it after use

        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {error_text}")
                
                result = await response.json()
                return result.get('response', '').strip()

        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return "0.0"

        finally:
            if close_after:
                await self._session.close()
                self._session = None  # Prevent further use of the closed session
        
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
        Text: "{ai_response}"
        """
        
        try:
            score_text = await self.llm.generate_response(prompt)
            logger.debug(f"Raw score response for {metric}: {score_text}")
            
            # Normalize and check if response is True or False
            score_text = score_text.strip().lower()
            if "true" in score_text or "yes" in score_text or score_text == "1":
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error generating {metric} score: {e}")
            return False  # Return default score on error
            
    async def evaluate(self, prompt: str, ai_response: str) -> Dict[str, Any]:
        """Evaluate the response and return the scores."""
        try:
            scores = await self.generate_scores(prompt, ai_response)
            return scores
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return self.get_default_evaluation()

    async def generate_scores(self, prompt: str, response: str) -> Dict[str, float]:
        """Generate scores for the response based on prompt."""
        try:
            thoughtful_score = int(await self._score_metric(response, "thoughtful"))
            helpful_score = int(await self._score_metric(response, "helpful"))
            intelligent_score = int(await self._score_metric(response, "intelligent"))
            nice_score = int(await self._score_metric(response, "nice"))
            kind_score = int(await self._score_metric(response, "kind"))
            
            overall_score = (thoughtful_score + helpful_score + intelligent_score + 
                           nice_score + kind_score) / 5.0

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
        print(f'Initializing FastKindnessAgent with model: {model_name}')
        self.model_name = model_name
        self.llm = OllamaLLM()
        self.evaluator = FastKindnessEvaluator(llm=self.llm)
        self.client = None

    async def __aenter__(self):
        """Set up async resources."""
        print("Setting up FastKindnessAgent resources...")
        self.client = edgedb.create_async_client()
        await self.llm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async resources."""
        print("Cleaning up FastKindnessAgent resources...")
        if self.client:
            await self.client.aclose()
        await self.llm.__aexit__(exc_type, exc_val, exc_tb)
        
    async def execute(self, prompt: str, ai_response: str) -> str:
        """Evaluate the ai_response and save the results."""
        start_time = time.time()
        try:
            evaluation = await self.evaluator.evaluate(prompt, response)
            await self.save_results(prompt, response, evaluation)

            # Log execution time
            total_time = round((time.time() - start_time) * 1000)
            logger.info(f"Total execution time: {total_time}ms")
            
            return json.dumps(evaluation)
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            return json.dumps(self.evaluator.get_default_evaluation())

    async def save_results(self, prompt: str, response: str, scores: Dict[str, float]):
        """Save results to JSONL and EdgeDB."""
        try:
            # Save to JSONL file
            await save_ai_output_to_jsonl(
                model_name=self.model_name,
                topic=prompt,
                content=response,
                jsonl_file_path="data/ai_outputs.jsonl",
                **scores
            )
            print(f'Saved to JSONL: {prompt} - {response}')
            
            if self.client:
                # Save to EdgeDB
                query = """
                INSERT AIOutput {
                    model_name := <str>$model_name,
                    topic := <str>$topic,
                    content := <str>$content,
                    thoughtful_score := <float64>$thoughtful_score,
                    helpful_score := <float64>$helpful_score,
                    intelligent_score := <float64>$intelligent_score,
                    nice_score := <float64>$nice_score,
                    kind_score := <float64>$kind_score,
                    overall_score := <float64>$overall_score
                }
                """
                await self.client.query(
                    query,
                    model_name=self.model_name,
                    topic=prompt,
                    content=response,
                    **scores
                )
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

# Example usage
async def main():
    """Main function demonstrating the usage of FastKindnessAgent."""
    print("Starting main function...")
    try:
        async with FastKindnessAgent() as agent:
            print("Agent initialized, executing test...")
            result = await agent.execute(
                prompt="How do I bake cookies?",
                response="Here's a detailed recipe..."
            )
            print(f"Execution result: {result}")
    except Exception as e:
        print(f"Error in main: {e}")
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    print("Starting script...")
    asyncio.run(main())
    print("Script completed.")