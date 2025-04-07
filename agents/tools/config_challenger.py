import os
from dataclasses import dataclass
from typing import Optional
import gel
from pathlib import Path
import json
from datetime import datetime
import pytz
import asyncio
from loguru import logger

# Ensure the 'data' folder exists
os.makedirs("data", exist_ok=True)


@dataclass
class Config:
    # Ollama settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "granite3.1-moe:1b")

    # Database settings
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5656")  # Default Gel database port

    # Application settings
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
    VERBOSE: bool = bool(os.getenv("VERBOSE", "True"))

    def __post_init__(self):
        if self.VERBOSE:
            logger.info(f"Configuration loaded: {self.__dict__}")


config = Config()


async def save_ai_output_to_jsonl(
    model_name: str,
    topic: str,
    content: str,
    thoughtful_score: Optional[float] = None,
    helpful_score: Optional[float] = None,
    intelligent_score: Optional[float] = None,
    nice_score: Optional[float] = None,
    kind_score: Optional[float] = None,
    overall_score: Optional[float] = None,
    jsonl_file_path: str = "data/ai_outputs.jsonl",
) -> None:
    """
    Asynchronously save AI output to both Gel database and JSONL file.

    Args:
        model_name: Name of the model used
        topic: Input prompt/topic
        content: Generated response
        *_score: Various evaluation scores
        jsonl_file_path: Path to save JSONL output
    """
    try:
        # Get the current time and make it timezone-aware
        timestamp = datetime.now(pytz.utc)

        # Create async Gel database client
        async with gel.create_async_client() as client:
            # Insert into Gel database database
            await client.query_single(
                """
                INSERT AIOutput {
                    model_name := <str>$model_name,
                    topic := <str>$topic,
                    content := <str>$content,
                    thoughtful_score := <optional float64>$thoughtful_score,
                    helpful_score := <optional float64>$helpful_score,
                    intelligent_score := <optional float64>$intelligent_score,
                    nice_score := <optional float64>$nice_score,
                    kind_score := <optional float64>$kind_score,
                    overall_score := <optional float64>$overall_score,
                    timestamp := <datetime>$timestamp
                }
            """,
                model_name=model_name,
                topic=topic,
                content=content,
                thoughtful_score=thoughtful_score,
                helpful_score=helpful_score,
                intelligent_score=intelligent_score,
                nice_score=nice_score,
                kind_score=kind_score,
                overall_score=overall_score,
                timestamp=timestamp,
            )

        # Prepare data for JSONL output
        jsonl_data = {
            "timestamp": timestamp.isoformat(),
            "model_name": model_name,
            "input": topic,
            "output": content,
            "scores": {
                "thoughtful_score": thoughtful_score,
                "helpful_score": helpful_score,
                "intelligent_score": intelligent_score,
                "nice_score": nice_score,
                "kind_score": kind_score,
                "overall_score": overall_score,
            },
        }

        # Asynchronously write to JSONL file
        # Since file I/O is blocking, we use asyncio.to_thread for non-blocking operation
        await asyncio.to_thread(_write_jsonl, jsonl_file_path, jsonl_data)

        if config.VERBOSE:
            logger.info(
                f"AI output saved successfully to Gel database and {jsonl_file_path}"
            )

    except Exception as e:
        logger.error(f"Error saving AI output: {e}")
        raise


def _write_jsonl(file_path: str, data: dict) -> None:
    """Helper function to write JSONL data synchronously."""
    try:
        with open(file_path, "a") as file:
            file.write(json.dumps(data) + "\n")
    except Exception as e:
        logger.error(f"Error writing to JSONL file: {e}")
        raise


# Gel database schema for reference:
"""
module default {
    type AIOutput {
        required property model_name -> str;
        required property topic -> str;
        required property content -> str;
        property thoughtful_score -> float64;
        property helpful_score -> float64;
        property intelligent_score -> float64;
        property nice_score -> float64;
        property kind_score -> float64;
        property overall_score -> float64;
        required property timestamp -> datetime;
    }
}
"""
