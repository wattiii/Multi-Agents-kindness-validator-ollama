import asyncio
import json
import time
from typing import Dict, Any, Optional
import logging
import aiohttp

from manage_database import DatabaseManager

# Standard logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OllamaLLM:
    """Custom Ollama LLM class for generating responses."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model: str = "gemma3:1b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 256,
    ):
        if config is None:
            config = {}
        self.config = config
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._session = None
        self._session_owner = False  # Track whether we created the session

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        self._session_owner = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session and self._session_owner:
            await self._session.close()
            self._session = None

    async def generate_response(self, prompt: str) -> Optional[str]:
        if not self._session:
            self._session = aiohttp.ClientSession()
            close_after = True
        else:
            close_after = False

        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }

            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {error_text}")
                    return None

                result = await response.json()
                logger.debug(f"Response JSON: {result}")
                return result.get("response", "").strip()

        except aiohttp.ClientError as e:
            logger.error(f"Network error calling Ollama API: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama API: {e}")
            return None
        finally:
            if close_after:
                await self._session.close()
                self._session = None


class FastKindnessEvaluator:
    """Evaluation module to assess AI responses."""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        if llm is None:
            llm = OllamaLLM(config={})
        self.llm = llm

    async def _score_metric(self, response: str, metric: str) -> bool:
        """Generic scoring method for any metric with debug logging."""
        prompt = (
            f"You are a thoughtful, helpful, intelligent, nice, and kind scoring assistant. "
            f"Rate the following text on how {metric} it is. "
            'If the response meets the standard, return "True". '
            'If the response does not meet the standard, return "False". '
            f'Text: "{response}"'
        )

        try:
            score_text = await self.llm.generate_response(prompt)
            print(f"[DEBUG] Raw LLM response for metric '{metric}': {score_text}")
            logger.debug(f"Raw score response for {metric}: {score_text}")

            if score_text:
                score_text = score_text.strip().lower()
                if "true" in score_text or "yes" in score_text or score_text == "1":
                    return True
            return False

        except Exception as e:
            logger.error(f"Error generating {metric} score: {e}")
            return False  # Return default score on error

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
            thoughtful_score = int(await self._score_metric(response, "thoughtful"))
            helpful_score = int(await self._score_metric(response, "helpful"))
            intelligent_score = int(await self._score_metric(response, "intelligent"))
            nice_score = int(await self._score_metric(response, "nice"))
            kind_score = int(await self._score_metric(response, "kind"))

            overall_score = (
                thoughtful_score
                + helpful_score
                + intelligent_score
                + nice_score
                + kind_score
            ) / 5.0

            return {
                "thoughtful_score": thoughtful_score,
                "helpful_score": helpful_score,
                "intelligent_score": intelligent_score,
                "nice_score": nice_score,
                "kind_score": kind_score,
                "overall_score": overall_score,
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
            "overall_score": 0.4444,
        }


class FastKindnessAgent:
    """Agent to interact with the model and save results."""

    def __init__(self, model_name: str = "gemma3:1b"):
        print(f"Initializing FastKindnessAgent with model: {model_name}")
        self.model_name = model_name
        self.evaluator = FastKindnessEvaluator()  # Use default evaluator
        self.db_manager = DatabaseManager()

    async def __aenter__(self):
        """No database setup needed here anymore."""
        print("FastKindnessAgent setup complete.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """No database cleanup needed here anymore."""
        print("FastKindnessAgent cleanup complete.")

    async def execute(self, prompt: str, response: str) -> str:
        """Evaluate the response and save the results."""
        start_time = time.time()
        try:
            logger.info(f"Starting evaluation for prompt: {prompt}")
            evaluation = await self.evaluator.evaluate(prompt, response)
            logger.info(f"Evaluation completed: {evaluation}")

            # Save results using DatabaseManager (saves to both JSONL and EdgeDB)
            await self.save_results(prompt, response, evaluation)

            total_time = round((time.time() - start_time) * 1000)
            logger.info(f"Total execution time: {total_time}ms")
            return json.dumps(evaluation)

        except Exception as e:
            logger.error(f"Error during execution: {e}")
            return json.dumps(self.evaluator.get_default_evaluation())

    async def save_results(self, prompt: str, response: str, scores: Dict[str, float]):
        """Save results to JSONL and EdgeDB."""
        try:
            await self.db_manager.save_results(prompt, response, scores)
            logger.info(f"Saved to JSONL and EdgeDB: {prompt} - {response}")
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
                prompt="example prompt",
                response="example response",
            )
            print(f"Execution result: {result}")
    except Exception as e:
        print(f"Error in main: {e}")
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    print("Starting script...")
    asyncio.run(main())
    print("Script completed.")
