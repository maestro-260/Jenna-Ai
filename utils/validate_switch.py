import asyncio
from utils.model_switcher import ModelSwitcher
from core.cognitive.reasoner import AdvancedReasoner


async def test_switch():
    switcher = ModelSwitcher()
    reasoner = AdvancedReasoner()
    
    print("Current model:", reasoner.llm)
    if await switcher.switch():
        await reasoner.reload_components()
        print("New model:", reasoner.llm)
        return True
    return False

if __name__ == "__main__":
    asyncio.run(test_switch())