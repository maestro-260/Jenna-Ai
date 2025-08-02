import asyncio
import logging
import sounddevice as sd
import json
import os
from vosk import Model, KaldiRecognizer
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class WakeWordDetector:
    def __init__(self, callback_handler: Optional[Callable] = None):
        self.callback_handler = callback_handler
        self.model_path = os.getenv("VOSK_MODEL", "models/vosk-model-small-en-us")
        self.wake_phrases = ["hey jenna", "hello jenna", "jenna"]
        self.audio_queue = asyncio.Queue()
        self.is_listening = False
        self.cooldown_seconds = 2.0
        self.last_detection_time = 0
        self.stream = None

        # Lazy loading of models
        self._model = None
        self._recognizer = None

    async def initialize(self):
        """Initialize the wake word detector asynchronously"""
        await self._load_model()
        return self

    async def _load_model(self):
        """Load the wake word detection model if not already loaded"""
        if self._model is None:
            logger.info(f"Loading wake word model from {self.model_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model path not found: {self.model_path}"
                    )
            self._model = Model(self.model_path)
            self._recognizer = KaldiRecognizer(self._model, 16000)
            logger.info("Wake word model loaded")

    def audio_callback(self, indata, frames, time, status):
        """Callback for the sounddevice stream"""
        if status:
            logger.error(f"Audio callback status: {status}")
        if self.is_listening:
            self.audio_queue.put_nowait(indata.tobytes())

    async def process_audio_chunk(self, chunk: bytes) -> Optional[str]:
        """Simplified wake word detection."""
        if not self._recognizer:
            await self._load_model()

        if self._recognizer.AcceptWaveform(chunk):
            result = json.loads(self._recognizer.Result())
            text = result.get("text", "").lower().strip()
            if any(phrase in text for phrase in self.wake_phrases):
                return text
        return None

    async def listen_for_wake_word(self):
        """Main detection loop"""
        self.is_listening = True

        # Load model before starting
        await self._load_model()

        try:
            with sd.RawInputStream(
                samplerate=16000,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self.audio_callback
            ) as stream:
                self.stream = stream
                logger.info("Wake word detection active")

                while self.is_listening:
                    try:
                        # Wait for audio data with timeout
                        chunk = await asyncio.wait_for(
                            self.audio_queue.get(), timeout=1.0
                        )

                        # Process the audio chunk
                        detected_text = await self.process_audio_chunk(chunk)

                        # Handle wake word detection
                        if detected_text:
                            current_time = asyncio.get_event_loop().time()
                            if (current_time - self.last_detection_time >
                                    self.cooldown_seconds):
                                logger.info(
                                    f"Wake phrase detected: '{detected_text}'"
                                    )
                                self.last_detection_time = current_time
                                await self.handle_wake_word_detected(
                                    detected_text
                                )

                    except asyncio.TimeoutError:
                        # Timeout is expected, just continue
                        continue
                    except Exception as e:
                        logger.error(f"Audio processing error: {e}")
                        await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
        finally:
            if self.stream:
                self.stream.close()
            logger.info("Wake word detector stopped listening")

    async def handle_wake_word_detected(self, detected_text: str):
        """
        Handle wake word detection

        Args:
            detected_text: The detected text containing the wake word
        """
        # Pause detection during interaction
        was_listening = self.is_listening
        self.is_listening = False

        try:
            # Call the handler if provided
            if self.callback_handler:
                await self.callback_handler(detected_text)

        except Exception as e:
            logger.error(f"Error in wake word handling: {e}")

        finally:
            # Resume listening after a brief pause
            await asyncio.sleep(1.0)
            self.is_listening = was_listening
            if was_listening:
                logger.info("Resuming wake word detection")

    async def start(self):
        """Start the wake word detection system"""
        try:
            await self.listen_for_wake_word()
        except Exception as e:
            logger.error(f"Failed to start wake word detector: {e}")
            raise

    async def stop(self):
        """Cleanly shutdown the detector"""
        self.is_listening = False
        if self.stream:
            self.stream.close()
        logger.info("Wake word detector stopped")

    async def pause(self):
        """Temporarily pause detection"""
        if self.is_listening:
            self.is_listening = False
            logger.info("Wake word detection paused")

    async def resume(self):
        """Resume detection after pause"""
        if not self.is_listening:
            self.is_listening = True
            logger.info("Wake word detection resumed")
