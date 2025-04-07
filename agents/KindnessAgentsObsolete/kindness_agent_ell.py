import ell
import aiohttp
from typing import Any, Dict
from pydantic import BaseModel, Field

# Assuming OllamaLLM is defined as before with the necessary methods
class OllamaLLM:
    """Custom Ollama LLM class for generating responses."""

    def __init__(self, model: str = "granite3.1-dense:2b", base_url: str = "http://localhost:11434", temperature: float = 0.2, max_tokens: int = 256):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._session = None
        self._session_owner = False  # Track whether we created the session

    async def generate_response(self, prompt: str) -> str:
        """Generate a response using Ollama's API."""
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()

            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }

            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {error_text}")

                result = await response.json()
                return result.get('response', '').strip()
        except Exception as e:
            return f"Error generating response: {e}"
        finally:
            if self._session:
                await self._session.close()

# Create the OllamaLLM client instance
ollama_client = OllamaLLM()

# Define the ell.simple decorated function
@ell.simple(model="granite3.1-dense:2b", client=ollama_client)
async def OllamaChat(prompt: str) -> str:
    """Function to interact with Ollama's API."""
    return await ollama_client.generate_response(prompt)






class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    rating: int = Field(description="The rating of the movie out of 10")
    summary: str = Field(description="A brief summary of the movie")

@ell.simple(model="gpt-3.5-turbo")
def generate_movie_review_manual(movie: str):
    return [
        ell.system(f"""You are a movie review generator. Given the name of a movie, you need to return a structured review in JSON format.

You must absolutely respond in this format with no exceptions.
{MovieReview.model_json_schema()}
"""),
        ell.user("Review the movie: {movie}"),
    ]

# parser support coming soon!
unparsed = generate_movie_review_manual("The Matrix")
parsed = MovieReview.model_validate_json(unparsed)




# Example usage
async def main():
    prompt = "How do I bake cookies?"
    response = await OllamaChat(prompt)
    print(response)

# Run the example
import asyncio
asyncio.run(main())
