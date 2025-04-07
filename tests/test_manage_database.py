import pytest
from unittest.mock import MagicMock, patch, call
from manage_database import DatabaseManager
import asyncio


@pytest.fixture
def db_manager():
    # Mock the client and pass it into the DatabaseManager
    mock_client = MagicMock()
    db_manager = DatabaseManager()
    db_manager.client = mock_client
    return db_manager


@pytest.mark.asyncio
async def test_save_results(db_manager):
    # Test that both EdgeDB and JSONL saving methods are called
    with patch("manage_database.aiofiles.open", new_callable=MagicMock) as mock_file:
        await db_manager.save_results(
            topic="Test Topic",
            content="Test Content",
            scores={
                "thoughtful_score": 4.5,
                "helpful_score": 4.2,
                "intelligent_score": 4.7,
                "nice_score": 4.8,
                "kind_score": 5.0,
                "overall_score": 4.6,
            },
        )

    # Verify that the EdgeDB save method is called
    db_manager.client.query_single.assert_called_once()

    # Verify that the JSONL file save method is called
    mock_file.assert_called_once()


@pytest.mark.asyncio
async def test_save_to_jsonl_error(db_manager):
    # Simulate error while writing to JSONL
    with patch(
        "manage_database.aiofiles.open", side_effect=Exception("File write error")
    ):
        try:
            await db_manager.save_to_jsonl(
                "Test Topic", "Test Content", {"score": 5}, "test.jsonl"
            )
        except Exception as e:
            assert str(e) == "File write error"

    # Ensure error handling works, but doesn't crash the program.
