import asyncio
import json
from datetime import datetime
import pytz
from agents.config import config  # This imports the global config instance
import aiofiles  # Add this import


class DatabaseManager:
    def __init__(self):
        self.client = (
            config.client
        )  # Use the shared Gel database client from the config

    def init_database(self):
        """Ensure the database schema exists"""
        try:
            self.create_schema()
            print("Database schema initialized successfully.")
        except Exception as e:
            print(f"Error initializing database: {e}")

    async def create_schema(self):
        """Create the necessary types and tables in the database."""
        existing_types = await self.client.query(
            """
            SELECT schema::ObjectType FILTER .name = 'default::AIOutput';
            """
        )

        if existing_types:
            print("AIOutput type already exists. Skipping creation.")
            return

        await self.client.query(
            """
            CREATE TYPE AIOutput {
                required property topic -> str;
                required property content -> str;
                property thoughtful_score -> float64;
                property helpful_score -> float64;
                property intelligent_score -> float64;
                property nice_score -> float64;
                property kind_score -> float64;
                property overall_score -> float64;
                required property timestamp -> datetime;
            };
            """
        )
        print("AIOutput type created.")

    def drop_schema(self):
        """Drop the schema if you need to reset the database."""
        try:
            self.client.query("DROP TYPE IF EXISTS AIOutput;")
            print("Schema dropped successfully.")
        except Exception as e:
            print(f"Error dropping schema: {e}")

    def reset_database(self):
        """Reset the database by dropping and re-creating the schema."""
        self.drop_schema()
        self.create_schema()
        print("Database has been reset.")

    async def save_results(
        self,
        topic: str,
        content: str,
        scores: dict,
        jsonl_file_path="data/ai_outputs.jsonl",
    ):
        """Save AI output to Gel database and append it to a JSONL file."""
        try:
            timestamp = datetime.utcnow().replace(tzinfo=pytz.utc)

            # Create async tasks for both actions
            tasks = [
                self.save_to_gel(topic, content, scores, timestamp),
                self.save_to_jsonl(topic, content, scores, jsonl_file_path),
            ]

            # Wait for both tasks to complete
            await asyncio.gather(*tasks)

        except Exception as e:
            print(f"Error saving results: {e}")

    async def save_to_gel(
        self, topic: str, content: str, scores: dict, timestamp: datetime
    ):
        """Save to Gel database asynchronously."""
        try:
            await self.client.query_single(
                """
                INSERT AIOutput {
                    topic := <str>$topic,
                    content := <str>$content,
                    thoughtful_score := <optional float64>$thoughtful_score,
                    helpful_score := <optional float64>$helpful_score,
                    intelligent_score := <optional float64>$intelligent_score,
                    nice_score := <optional float64>$nice_score,
                    kind_score := <optional float64>$kind_score,
                    overall_score := <optional float64>$overall_score,
                    timestamp := <datetime>$timestamp
                };
                """,
                topic=topic,
                content=content,
                thoughtful_score=scores.get("thoughtful_score"),
                helpful_score=scores.get("helpful_score"),
                intelligent_score=scores.get("intelligent_score"),
                nice_score=scores.get("nice_score"),
                kind_score=scores.get("kind_score"),
                overall_score=scores.get("overall_score"),
                timestamp=timestamp,
            )

            print(f"Saved to Gel database: {topic}")
        except Exception as e:
            print(f"Error saving to Gel database: {e}")

    async def save_to_jsonl(
        self, topic: str, content: str, scores: dict, jsonl_file_path: str
    ):
        """Append to JSONL asynchronously using aiofiles."""
        try:
            jsonl_data = {"input": topic, "output": content, "scores": scores}

            async with aiofiles.open(jsonl_file_path, "a") as file:
                await file.write(json.dumps(jsonl_data) + "\n")  # ← Use await here_

            print(f"AI output saved to JSONL: {jsonl_data}")
        except Exception as e:
            print(f"Error saving to JSONL: {e}")


# Example usage
async def main():
    db_manager = DatabaseManager()  # No need to pass client, it uses the global one
    await db_manager.save_results(
        topic="How do I bake cookies?",
        content="Here's a detailed recipe...",
        scores={
            "thoughtful_score": 1,
            "helpful_score": 1,
            "intelligent_score": 1,
            "nice_score": 0,
            "kind_score": 0,
            "overall_score": 0.6,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
