import asyncio
import json
import logging
from time import sleep
from duckduckgo_search import DDGS
import requests
from ollama import chat
from agents.agent_base import AgentBase
from agents.config import Config
from concurrent.futures import ThreadPoolExecutor
import time
from cachetools import TTLCache
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient(AgentBase):
    def __init__(self, name="OllamaAgent", max_retries=2, verbose=True):
        super().__init__(name, max_retries, verbose)
        self._model_name = Config.MODEL_NAME
        self._cache = TTLCache(maxsize=100, ttl=150)  # Cache with TTL of 2.5 minutes

    async def execute(self, question: str) -> str:
        """
        Generate a detailed explanation with proper research data.
        """
        research_data = await self.use_tools(question)
        explanation = await self.call_llama(question, research_data)
        return explanation

    async def use_tools(self, prompt: str) -> dict:
        """
        Fetch research data from DuckDuckGo search, philosophy quote, and AdviceSlip API concurrently, and measure execution time in milliseconds.
        """
        start_time = time.perf_counter()  # Start timer

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_ddg = executor.submit(self._duckduckgo_search, prompt)
            future_advice = executor.submit(self.get_general_advice)
            future_philosophy = executor.submit(self.get_philosophy_quote)

            # Wait for results
            result1 = future_ddg.result()
            result2 = future_advice.result()
            result3 = future_philosophy.result()

        # Print the results using f-string
        print(f"{[result1, result2, result3]}")

        end_time = time.perf_counter()  # End timer
        elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

        research_data = {
            "duckduckgo_results": (
                result1
                if isinstance(result1, list)
                else ["No DuckDuckGo results found."]
            ),
            "general_advice": (
                [result2] if isinstance(result2, str) else ["No advice available."]
            ),
            "philosophy_quotes": (
                [result3]
                if isinstance(result3, str)
                else ["No philosophy quotes available."]
            ),
        }

        logger.info(f"Output of use_tools:\n{json.dumps(research_data, indent=2)}")
        logger.info(f"🕒 use_tools execution time: {elapsed_time_ms:.2f} ms")

        return research_data

    def _duckduckgo_search(self, query: str, retries: int = 1, delay: int = 10) -> list:
        """
        Perform a search using DuckDuckGo and return the top results.
        """
        if query in self._cache:
            return self._cache[query]

        ddgs = DDGS()
        try:
            results = list(ddgs.text(keywords=query, max_results=2))
            self._cache[query] = results
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            if retries > 0:
                sleep(delay)
                return self._duckduckgo_search(query, retries - 1, delay * 2)
            return []

    def get_general_advice(self) -> str:
        """
        Fetch a general piece of advice from the AdviceSlip API.
        """
        try:
            response = requests.get("https://api.adviceslip.com/advice")
            advice = response.json()
            return advice.get("slip", {}).get("advice", "No advice available.")
        except Exception as e:
            logger.warning(f"Could not retrieve advice: {str(e)}")
            return "No advice available."

    def get_philosophy_quote(self) -> str:
        """
        Fetch a random philosophy quote from the Stoic Quotes API.
        """
        url = "https://stoic-quotes.com/api/quote"

        try:
            response = requests.get(
                url, timeout=5
            )  # Set a timeout to avoid hanging requests
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx, 5xx)
            data = response.json()  # Expecting a dictionary, not a list

            # Ensure data has expected structure
            if isinstance(data, dict) and "text" in data and "author" in data:
                return f'"{data["text"]}" — {data["author"]}'
            return "No philosophy quote available."
        except (
            requests.RequestException,
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
            requests.exceptions.JSONDecodeError,
        ) as e:
            logger.warning(f"Could not retrieve philosophy quote: {str(e)}")
            return "No philosophy quote available."

    async def call_llama(
        self,
        question_text: str,
        research_data: dict,
        model: Optional[str] = "gemma3:1b",
    ) -> str:
        """
        Call the Ollama model to generate a response incorporating the research data.
        """
        pass  # Placeholder implementation to avoid indentation error
        if model is None:
            model = "gemma3:1b"

        # Ensure research_data is a dictionary
        if not isinstance(research_data, dict):
            try:
                research_data = json.loads(research_data)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse research_data string: {research_data}")
                research_data = {}

        duckduckgo_results = research_data.get("duckduckgo_results", [])
        general_advice = research_data.get("general_advice", "No advice found.")
        philosophy_quotes = research_data.get("philosophy_quotes", [])

        # Make the call to Ollama's API
        response = chat(
            model=model,
            messages=[
                self._create_message_content(
                    question_text, duckduckgo_results, general_advice, philosophy_quotes
                )
            ],
        )

        # Extract response text safely
        return (
            response["message"]["content"]
            if "message" in response
            else "No response from Ollama."
        )

    def _create_message_content(
        self,
        question_text: str,
        duckduckgo_results: list,
        general_advice: str,
        philosophy_quotes: list,
    ) -> dict:
        """
        Create the message content for the Ollama API call.
        """
        content = (
            f"You are a thoughtful AI model that provides helpful and intelligent responses in a way that is thoughtful, helpful, intelligent, nice, and kind. "
            f"Read the user's question, research results, summarize, and provide a coherent thoughtful, helpful, intelligent, nice, and kind response. "
            f"The user's question is: {question_text}. Use the research results to provide a thoughtful, helpful, intelligent, nice, and kind summarized response. "
            f"The research results are: DuckDuckGo Results: {duckduckgo_results}, General Advice: {general_advice}, Philosophy Quotes: {philosophy_quotes}"
        )
        return {
            "role": "user",
            "content": content,
        }

    async def generate_response(self, input_text: str) -> str:
        """
        Simulate an asynchronous operation.
        """
        await asyncio.sleep(1)  # Simulate a delay
        return "This is a test response."
