import pytest
from services.main_service import JENNA
import torchaudio
import numpy as np


@pytest.fixture
def sample_audio():
    # Generate 1 sec of silence for testing
    sample_rate = 16000
    audio = np.zeros(sample_rate, dtype=np.float32)
    return audio, sample_rate


@pytest.mark.asyncio
async def test_full_conversation_flow():
    ai = JENNA()
    waveform, sample_rate = torchaudio.load("path/to/sample_audio.wav")
# Replace "path/to/sample_audio.wav" with the actual path to real audio sample.
    transcription = ai.audio.transcribe(waveform.numpy(), sample_rate)
    response = await ai.reasoner.process_query(transcription['text'], {})
    synthesized = ai.audio.synthesize(response['text'])
    assert len(transcription['text']) > 0
    assert 'response' in response
    assert isinstance(synthesized, np.ndarray)
    assert len(synthesized) > 16000  # Ensure audio is generated


@pytest.mark.asyncio
async def test_conversation_flow_with_sample_audio(sample_audio):
    ai = JENNA()
    audio, sample_rate = sample_audio
    transcription = ai.audio.transcribe(audio, sample_rate)
    assert isinstance(transcription, dict)