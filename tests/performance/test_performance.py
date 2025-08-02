import pytest
import time
import asyncio
from services.main_service import JENNA

@pytest.mark.performance
@pytest.mark.asyncio
async def test_response_time():
    jenna = JENNA()
    await jenna.initialize()
    
    start_time = time.time()
    response = await jenna._handle_interaction({
        "text": "Hello, how are you?",
        "emotion": "neutral"
    })
    end_time = time.time()
    
    response_time = end_time - start_time
    assert response_time < 2.0  # Response should be under 2 seconds

@pytest.mark.performance
@pytest.mark.asyncio
async def test_memory_usage():
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    jenna = JENNA()
    await jenna.initialize()
    
    # Run 100 interactions
    for _ in range(100):
        await jenna._handle_interaction({
            "text": "Test message",
            "emotion": "neutral"
        })
    
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
    
    assert memory_increase < 500  # Memory increase should be less than 500MB
