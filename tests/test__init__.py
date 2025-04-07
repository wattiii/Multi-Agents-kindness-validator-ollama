import pytest
from unittest.mock import patch, MagicMock
import asyncio
from agents.agent_base import AgentBase
import logging


# Dummy classes for testing
class DummyOllamaClient:
    def __init__(self):
        pass


class DummyWorkerAgent:
    def __init__(self, agent):
        self.agent = agent

    def execute(self, topic, outline=None):
        return f"content for {topic}"


class DummyOllamaLLM:
    def __init__(self, config):
        self.config = config

    def execute(self, topic, content):
        return f"evaluation for {topic} with content: {content}"


class DummyFastKindnessEvaluator:
    def __init__(self, llm):
        self.llm = llm

    async def execute(self, topic, content):
        await asyncio.sleep(0.01)
        return f"async evaluation for {topic} with content: {content}"


# Monkeypatch the agent classes in your module
@pytest.fixture(autouse=True)
def patch_agents(monkeypatch):
    monkeypatch.setattr("agents.WorkerAgent", DummyWorkerAgent)
    monkeypatch.setattr("agents.FastKindnessEvaluator", DummyFastKindnessEvaluator)
    monkeypatch.setattr("agents.OllamaLLM", DummyOllamaLLM)
    monkeypatch.setattr("agents.OllamaClient", DummyOllamaClient)
    logging.disable(logging.CRITICAL)


@pytest.fixture
def agent_base():
    config = {"dummy_key": "dummy_value"}
    return AgentBase(config, max_retries=2, verbose=False)


# Tests
def test_get_agent_success(agent_manager):
    worker_agent = agent_manager.get_agent("writer")
    evaluator_agent = agent_manager.get_agent("evaluator")
    assert writer_agent is not None
    assert evaluator_agent is not None


def test_get_agent_invalid(agent_manager):
    with pytest.raises(ValueError):
        agent_manager.get_agent("invalid_agent")


@pytest.mark.asyncio
async def test_execute_pipeline(agent_manager):
    async def mock_pipeline():
        return "success"

    with patch(
        "agents.pipeline", new=mock_pipeline
    ):  # Patch the pipeline function from _**_init_**_.py_
        result = await agent_manager.execute_pipeline("test_topic")
        assert result["status"] == "success"
        assert result["content"] == "content for test_topic"
        assert (
            result["evaluation"]
            == "evaluation for test_topic with content: content for test_topic"
        )


def test_get_available_agents(agent_manager):
    available = agent_manager.get_available_agents()
    assert "writer" in available
    assert "evaluator" in available


# Streamlit Integration Fix
def initialize_session_state():
    if "agent_manager" not in st.session_state:
        config = {"dummy_key": "dummy_value"}
        st.session_state.agent_manager = AgentManager(
            config, max_retries=2, verbose=False
        )


def main():
    initialize_session_state()
    st.write("Agent Manager Initialized")
    # Additional Streamlit UI logic here


if __name__ == "__main__":
    asyncio.run(main())
