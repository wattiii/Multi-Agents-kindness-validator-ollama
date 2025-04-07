import pytest
from unittest.mock import MagicMock
from agents.agent_base import AgentBase
from agents.worker import WorkerAgent
import asyncio


# Create a mock subclass of AgentBase for testing
class MockAgent(AgentBase):
    def __init__(self, name: str):
        super().__init__(name)

    async def execute(self, input_data: str) -> str:
        return f"Processed: {input_data}"


def test_worker_agent_initialization_valid():
    mock_agent = MockAgent(name="TestAgent")
    worker = WorkerAgent(mock_agent)
    assert worker._agent == mock_agent  # Use _agent instead of agent


def test_worker_agent_initialization_invalid():
    with pytest.raises(TypeError):
        worker = WorkerAgent("NotAnAgent")


def test_execute_valid_input():
    mock_agent = MockAgent(name="TestAgent")
    worker = WorkerAgent(mock_agent)
    response = asyncio.run(
        worker.execute("Test Input")
    )  # Use asyncio.run to run the coroutine
    assert response == "Processed: Test Input"


def test_execute_with_exception():
    mock_agent = MagicMock(spec=AgentBase)
    mock_agent.execute.return_value = Exception("Test Exception")
    worker = WorkerAgent(mock_agent)
    try:
        asyncio.run(
            worker.execute("Test Input")
        )  # Use asyncio.run to run the coroutine
    except Exception as e:
        assert (
            str(e)
            == "An error occurred during task execution of worker: Test Exception"
        )
