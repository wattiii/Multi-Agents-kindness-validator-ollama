import pytest
import aiohttp
import time
import json
import logging
from agents.ollama_client import OllamaClient
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture
def client():
    return OllamaClient()


@patch("agents.ollama_client.DDGS.text")
def test_duckduckgo_search(mock_ddgs, client):
    mock_ddgs.return_value = ["Result 1", "Result 2"]
    results = client._duckduckgo_search("test query")
    assert results == ["Result 1", "Result 2"]


@patch("requests.get")
def test_get_general_advice(mock_get, client):
    mock_get.return_value.json.return_value = {"slip": {"advice": "Test Advice"}}
    result = client.get_general_advice()
    assert result == "Test Advice"


@patch("requests.get")
def test_get_philosophy_quote(mock_get, client):
    mock_get.return_value.json.return_value = {
        "text": "Test Quote",
        "author": "Test Author",
    }
    result = client.get_philosophy_quote()
    assert result == '"Test Quote" — Test Author'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@patch("agents.ollama_client.OllamaClient._duckduckgo_search")
@patch("agents.ollama_client.OllamaClient.get_general_advice")
@patch("agents.ollama_client.OllamaClient.get_philosophy_quote")
async def test_use_tools(mock_philosophy, mock_advice, mock_ddg, client):
    # Mock the results of the methods
    mock_ddg.return_value = ["DDG Result"]
    mock_advice.return_value = "General Advice"
    mock_philosophy.return_value = "Philosophy Quote"

    # Call the use_tools method
    result = await client.use_tools("test prompt")

    # Assert the expected result
    assert result == {
        "duckduckgo_results": ["DDG Result"],
        "general_advice": ["General Advice"],
        "philosophy_quotes": ["Philosophy Quote"],
    }


if __name__ == "__main__":
    pytest.main()
