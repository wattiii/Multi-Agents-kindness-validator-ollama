import asyncio
import pytest
from agents.agent_base import AgentBase  # Import the AgentBase class
import ollama

# Define a dummy subclass of AgentBase for testing purposes.
class DummyAgent(AgentBase):
    async def execute(self, _args, **kwargs):
        # Not needed for testing call_model.
        pass

@pytest.fixture
def dummy_agent(monkeypatch):
    # Patch ollama.chat to simulate a successful response.
    def dummy_chat(model, messages):
        return {"message": {"content": "dummy reply"}}

    monkeypatch.setattr(ollama, "chat", dummy_chat)
    
    # Create an instance of DummyAgent.
    return DummyAgent(name="dummy", max_retries=2, verbose=False)

@pytest.mark.asyncio
async def test_call_model_success(dummy_agent):
    model_name = "dummy-model"
    messages = [{"role": "user", "content": "Hello"}]
    reply = await dummy_agent.call_model(model_name, messages)
    assert reply == "dummy reply"

@pytest.mark.asyncio
async def test_call_model_failure(monkeypatch):
    # Simulate a failing model call that always raises an exception.
    call_count = 0

    def failing_chat(model, messages):
        nonlocal call_count
        call_count += 1
        raise Exception("failed")

    monkeypatch.setattr(ollama, "chat", failing_chat)
    agent = DummyAgent(name="dummy", max_retries=2, verbose=False)
    model_name = "dummy-model"
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(Exception, match="failed"):
        await agent.call_model(model_name, messages)

    # Verify that it retried at least max_retries times.
    assert call_count >= agent.max_retries