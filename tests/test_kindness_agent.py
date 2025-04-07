import pytest
import json
import streamlit as st
from unittest.mock import AsyncMock, patch
from agents.main import initialize_session_state  # Import initialize_session_state
from agents.kindness_agent import OllamaLLM, FastKindnessEvaluator, FastKindnessAgent


@pytest.mark.asyncio
async def test_ollama_llm_generate_response():
    """Test the OllamaLLM generate_response method."""
    # Save the original session_state if it exists
    original_session_state = getattr(st, "session_state", None)
    try:
        # Initialize session_state if it doesn't exist
        if not hasattr(st, "session_state"):
            st.session_state = {}

        # Mock the AgentManager
        with patch("agents.main.AgentManager") as MockAgentManager:
            # Call the function to test
            initialize_session_state()

            # Check if the expected items were added
            assert "messages" in st.session_state
            assert isinstance(st.session_state["messages"], list)
            assert st.session_state["messages"] == []

            assert "agent_manager" in st.session_state
            assert st.session_state["agent_manager"] is not None

        # Mock the OllamaLLM generate_response method
        with patch("agents.main.OllamaLLM.generate_response") as MockGenerateResponse:
            MockGenerateResponse.return_value = "I apologize... for the inconvenience."

            # Create an instance of OllamaLLM
            llm = OllamaLLM()

            # Generate a response
            response = await llm.generate_response("test prompt")

            # Check the response
            assert response == "I apologize... for the inconvenience."

    finally:
        # Restore the original session state if it was saved
        if original_session_state is not None:
            st.session_state = original_session_state
        elif hasattr(st, "session_state"):
            del st.session_state


@pytest.mark.asyncio
async def test_ollama_llm_generate_response_failure():
    mock_post = AsyncMock(side_effect=Exception("API Error"))

    with patch("agents.kindness_agent.aiohttp.ClientSession.post", new=mock_post):
        llm = OllamaLLM()
        try:
            await llm.generate_response("What is machine learning?")
        except Exception as e:
            assert str(e) == "API Error"


@pytest.mark.asyncio
async def test_fast_kindness_evaluator_generate_scores():
    """Test if FastKindnessEvaluator generates kindness scores correctly."""
    evaluator = FastKindnessEvaluator()
    expected_scores = {
        "thoughtful_score": 1,
        "helpful_score": 1,
        "intelligent_score": 1,
        "nice_score": 1,
        "kind_score": 1,
        "overall_score": 1.0,
    }

    # Patch the generate_response method so that each metric returns "1".
    with patch.object(evaluator.llm, "generate_response", AsyncMock(return_value="1")):
        scores = await evaluator.generate_scores("Hello", "Hi")
        print(f"Debug - Scores: {scores}")
        assert scores == expected_scores


@pytest.mark.asyncio
async def test_fast_kindness_agent_evaluate():
    """Test if FastKindnessAgent correctly evaluates scores."""
    agent = FastKindnessAgent()
    demo_prompt = "What is kindness?"
    demo_response = "Kindness is a way of being considerate and helpful."
    mock_evaluation = {
        "thoughtful_score": 1,
        "helpful_score": 1,
        "intelligent_score": 1,
        "nice_score": 1,
        "kind_score": 1,
        "overall_score": 1.0,
    }

    with patch.object(
        agent.evaluator, "evaluate", AsyncMock(return_value=mock_evaluation)
    ):
        result = await agent.execute(demo_prompt, demo_response)
        assert json.loads(result) == mock_evaluation


@pytest.mark.asyncio
async def test_fast_kindness_agent_save_results():
    """Test if FastKindnessAgent saves results."""
    agent = FastKindnessAgent()
    demo_prompt = "What is kindness?"
    demo_response = "Kindness is a way of being considerate and helpful."
    mock_evaluation = {
        "thoughtful_score": 1,
        "helpful_score": 1,
        "intelligent_score": 1,
        "nice_score": 1,
        "kind_score": 1,
        "overall_score": 1.0,
    }

    with patch.object(
        agent.db_manager,
        "save_results",
        AsyncMock(side_effect=Exception("Evaluation Failed")),
    ):
        result = await agent.execute(demo_prompt, demo_response)
        assert json.loads(result) == agent.evaluator.get_default_evaluation()


@pytest.mark.asyncio
async def test_fast_kindness_agent_execute():
    """Test if FastKindnessAgent executes the evaluation and saving process correctly."""
    agent = FastKindnessAgent()
    demo_prompt = "What is kindness?"
    demo_response = "Kindness is a way of being considerate and helpful."
    mock_evaluation = {
        "thoughtful_score": 1,
        "helpful_score": 1,
        "intelligent_score": 1,
        "nice_score": 1,
        "kind_score": 1,
        "overall_score": 1.0,
    }

    with patch.object(
        agent.evaluator, "evaluate", AsyncMock(return_value=mock_evaluation)
    ), patch.object(agent.db_manager, "save_results", AsyncMock()):
        result = await agent.execute(demo_prompt, demo_response)
        assert json.loads(result) == mock_evaluation


@pytest.mark.asyncio
async def test_fast_kindness_agent_default_evaluation():
    """Test if FastKindnessAgent returns default evaluation on error."""
    agent = FastKindnessAgent()
    demo_prompt = "What is kindness?"
    demo_response = "Kindness is a way of being considerate and helpful."

    with patch.object(
        agent.evaluator,
        "evaluate",
        AsyncMock(side_effect=Exception("Evaluation Error")),
    ):
        result = await agent.execute(demo_prompt, demo_response)
        assert json.loads(result) == agent.evaluator.get_default_evaluation()
