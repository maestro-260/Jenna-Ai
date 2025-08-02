import pytest
from core.cognitive.reasoner import AdvancedReasoner

@pytest.mark.asyncio
async def test_process_query():
    reasoner = AdvancedReasoner()
    response = await reasoner.process_query("What is the weather today?")
    assert "text" in response
    assert len(response["text"]) > 0
