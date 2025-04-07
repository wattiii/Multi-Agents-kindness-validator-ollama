import asyncio
from kindness_agent import OllamaLLM

async def test_ollama_request():
    async with OllamaLLM() as llm:
        response = await llm.generate_response("What is your name?")
        print(response)

asyncio.run(test_ollama_request())