from agents.worker import WorkerAgent
from agents.kindness_agent import FastKindnessEvaluator, OllamaLLM
from typing import Dict, Optional
import logging
from agents.ollama_client import OllamaClient
import asyncio


class AgentManager:
    def __init__(self, config: dict, max_retries: int = 2, verbose: bool = True):
        """
        Initializes the AgentManager with logging, agents, and configuration options.
        """
        self._logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)
        self._config = config  # Store configuration if needed

        try:
            self._agents: Dict[str, object] = {
                "writer": WorkerAgent(agent=OllamaClient()),
                "evaluator": FastKindnessEvaluator(OllamaLLM(self._config)),
            }
            self._logger.info("Agents initialized successfully")
        except Exception as e:
            self._logger.error(f"Error initializing agents: {str(e)}")
            raise

    def get_agent(self, agent_name: str) -> object:
        """
        Retrieves an agent by name, returning None if it does not exist.
        """
        agent = self._agents.get(agent_name)
        if not agent:
            self._logger.warning(f"Agent '{agent_name}' not found")
        return agent

    async def execute_pipeline(self, topic: str, outline: Optional[str] = None) -> dict:
        """
        Executes the content creation and evaluation pipeline asynchronously.
        """
        try:
            writer = self.get_agent("writer")
            if not writer:
                raise ValueError("Writer agent not found")

            content = await writer.execute(topic, outline)

            evaluator = self.get_agent("evaluator")
            if not evaluator:
                raise ValueError("Evaluator agent not found")

            evaluation = await evaluator.execute(topic, content)

            return {
                "topic": topic,
                "content": content,
                "evaluation": evaluation,
                "status": "success",
            }
        except Exception as e:
            self._logger.error(f"Pipeline execution failed: {str(e)}")
            return {"topic": topic, "error": str(e), "status": "failed"}

    def execute_pipeline_sync(self, topic: str, outline: Optional[str] = None) -> dict:
        """
        Synchronous wrapper for `execute_pipeline`, avoiding multiple event loops.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._logger.warning(
                    "Async event loop is already running; using `asyncio.ensure_future` instead."
                )
                return loop.run_until_complete(self.execute_pipeline(topic, outline))
            else:
                return asyncio.run(self.execute_pipeline(topic, outline))
        except RuntimeError:
            self._logger.error("Error executing pipeline in a sync environment.")
            return {
                "topic": topic,
                "error": "Async execution error",
                "status": "failed",
            }

    def get_available_agents(self) -> list:
        """
        Returns a list of available agent names.
        """
        return list(self._agents.keys())


async def main():
    client = OllamaClient()
    research_data = await client.use_tools("example prompt")
    print(research_data)


asyncio.run(main())
