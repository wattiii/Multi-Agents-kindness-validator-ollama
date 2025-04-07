# tests/test_main.py
import pytest
import streamlit as st
from agents.main import initialize_session_state
from unittest.mock import patch


@pytest.mark.asyncio
async def test_initialize_session_state():
    """Test the initialize_session_state function."""
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

    finally:
        # Restore the original session state if it was saved
        if original_session_state is not None:
            st.session_state = original_session_state
        elif hasattr(st, "session_state"):
            del st.session_state


if __name__ == "__main__":
    pytest.main()
