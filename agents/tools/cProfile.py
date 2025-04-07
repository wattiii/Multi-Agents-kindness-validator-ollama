import asyncio
import cProfile
import pstats
import io
from ..kindness_agent import FastKindnessAgent


def profile_function(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        print(s.getvalue())
        return result

    return wrapper


@profile_function
async def main():
    agent = FastKindnessAgent()
    result = await agent.execute(prompt="example prompt", response="example response")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
