from agents.agent_base import AgentBase
import asyncio
import time
import logging


class WorkerAgent:
    def __init__(self, agent: AgentBase):
        """
        Initialize the Worker with an agent (which is a subclass of AgentBase).

        :param agent: An instance of a class that inherits from AgentBase (e.g., OllamaClient)
        """
        if not isinstance(agent, AgentBase):
            raise TypeError("agent must be an instance of AgentBase or its subclass.")
        self._agent = agent
        self._logger = logging.getLogger(__name__)

    async def execute(self, response: str) -> str:
        """
        Performs the task by invoking the agent's execution logic.

        This method interacts with the agent's `execute()` method, passing the correct answer,
        and returns the result of the agent's task.

        :param response: The correct answer that the agent will process.
        :return: The result returned by the agent's execute method (e.g., explanation, response).
        """
        start_time = time.perf_counter()  # Start timer
        try:
            # Call the execute method of the agent and return its response
            result = await self._agent.execute(response)
            end_time = time.perf_counter()  # End timer
            elapsed_time_ms = (end_time - start_time) * 1000
            self._logger.info(f"🕒 Worker execution time: {elapsed_time_ms:.2f} ms")
            return result
        except Exception as e:
            # Handle any exception that may occur during task execution
            self._logger.error(
                f"An error occurred during task execution of worker: {str(e)}"
            )
            return f"An error occurred during task execution of worker: {str(e)}"
