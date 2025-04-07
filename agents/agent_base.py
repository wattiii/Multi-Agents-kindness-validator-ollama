from abc import ABC, abstractmethod
from loguru import logger
import ollama
import asyncio


class AgentBase(ABC):
    def __init__(self, name: str, max_retries: int = 2, verbose: bool = True):
        """
        The function initializes an object with a name, maximum number of retries, and a verbosity flag.

        :param name: The name of the agent.
        :param max_retries: The maximum number of retries allowed for a certain operation or task.
        :param verbose: A boolean parameter to control whether the object should display detailed output or not.
        """
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    async def execute(self, *args, **kwargs) -> None:
        """Abstract async execute method that must be implemented in subclasses."""
        pass

    async def call_model(
        self,
        model_name: str,
        messages: list,
        provider: str = "ollama",
        temperature: float = 0.4,
    ) -> str:
        """
        Calls a model asynchronously by running the blocking `ollama.chat` in a separate thread.

        :param model_name: Name of the model to use.
        :param messages: A list of message dictionaries.
        :param provider: The LLM provider to use (default is 'ollama').
        :param temperature: Sampling temperature.
        :return: The content of the model's response.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                if self.verbose:
                    logger.info(
                        f"[{self.name}] Sending messages to model: {model_name}"
                    )
                    for msg in messages:
                        logger.debug(f" {msg['role']}: {msg['content']}")

                # Use asyncio.to_thread() to run the blocking function in a separate thread
                response = await asyncio.to_thread(
                    ollama.chat,
                    model=model_name,
                    messages=messages,
                )

                # Parse the response to extract the text content
                reply = response["message"]["content"]

                if self.verbose:
                    logger.info(f"[{self.name}] Received response: {reply}")

                return reply
            except Exception as e:
                retries += 1
                logger.error(
                    f"[{self.name}] Error in agent_base during model call: {e}. Retry {retries}/{self.max_retries}"
                )

        raise Exception(
            f"[{self.name}] Agent_base failed to get response after {self.max_retries} retries."
        )
