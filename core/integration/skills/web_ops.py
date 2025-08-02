from browser_use import Agent, Browser, BrowserConfig
from langchain_community.llms import Ollama
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebOperationError(Exception):
    pass


class WebOperator:
    def __init__(self, llm_model="mistral"):
        # Initialize local LLM with Ollama
        self.llm = Ollama(
            model=llm_model,
            temperature=0.7,
            base_url='http://localhost:11434'
        )
        
        # Configure browser with enhanced settings
        self.browser = Browser(
            config=BrowserConfig(
                headless=True,
                stealth=True,
                user_agent=(
                    (
                        (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/119.0.0.0 Safari/537.36"
                        )
                    )
                ),
                block_resources=True,
                cache_enabled=False
            )
        )
        
        # Create agent with local LLM
        self.agent = Agent(llm=self.llm, browser=self.browser)
        logger.info(f"Initialized WebOperator with {llm_model} LLM")

    async def execute(self, task: dict) -> dict:
        try:
            if task["action"] == "search":
                return await self._web_search(task["query"])
            elif task["action"] == "scrape":
                return await self._scrape_data(
                    task["url"], task.get("selectors", [])
                )
            elif task["action"] == "automate":
                return await self._multi_step_flow(task["steps"])
            else:
                raise WebOperationError(
                    f"Unsupported action: {task['action']}"
                )
        except Exception as e:
            logger.error(f"Web operation failed: {e}")
            return {"error": str(e)}  # Return an error instead of raising it

    async def _web_search(self, query: str) -> dict:
        result = await self.agent.run_task(f"Search for '{query}' on Google")
        return {"results": result[:3]}

    async def _scrape_data(self, url: str, selectors: list) -> dict:
        await self.agent.run_task(f"Navigate to {url}")
        scraped = [
            await self.agent.run_task(f"Extract {s}") 
            for s in selectors
        ]
        return {"scraped_data": scraped}

    async def _multi_step_flow(self, steps: list) -> dict:
        for step in steps:
            await self.agent.run_task(
                f"Perform {step['action']} on {step['target']}"
            )
            await asyncio.sleep(1)
        return {"status": "success"}