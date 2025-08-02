import pytest
import asyncio
from services.main_service import JENNA
from core.cognitive.reasoner import AdvancedReasoner
from core.perception.audio_service import AudioProcessor

@pytest.mark.asyncio
async def test_system_initialization():
    jenna = JENNA()
    await jenna.initialize()
    assert jenna.initialized
    assert jenna.reasoner is not None
    assert jenna.audio is not None

@pytest.mark.asyncio
async def test_complete_interaction():
    jenna = JENNA()
    await jenna.initialize()
    
    # Test text input processing
    response = await jenna._handle_interaction({
        "text": "What's the weather like today?",
        "emotion": "neutral"
    })
    assert "text" in response
    assert "emotion" in response
    
    # Test audio input processing
    audio_input = await jenna._capture_audio_input()
    assert isinstance(audio_input, dict)
    assert "text" in audio_input

@pytest.mark.asyncio
async def test_error_scenarios():
    jenna = JENNA()
    await jenna.initialize()
    
    # Test invalid input handling
    response = await jenna._handle_interaction({
        "text": "",
        "emotion": "neutral"
    })
    assert "error" in response

    # Test API failure handling
    response = await jenna._handle_interaction({
        "text": "trigger_api_error",
        "emotion": "neutral"
    })
    assert response.get("text") != ""
