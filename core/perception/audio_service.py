import asyncio
import torchaudio
import numpy as np
import sounddevice as sd
import torch
import gc
import os
import logging
from typing import Dict, Optional
from faster_whisper import WhisperModel
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class AudioProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        # Audio config
        self.sample_rate = 24000  # OpenVoice uses 24kHz
        self.audio_buffer = np.array([])
        # store 16kHz audio for prosody analysis

        # Lazy-loaded components (initialized when needed)
        self._stt_model = None
        self._tts_model = None

        # Initialize tone color converter first (needed for voice loading)
        self.tone_color_converter = ToneColorConverter(device=self.device)
        self.reference_se = None  # Will be loaded on demand

    def _load_reference_voice(self):
        """Load reference voice style embedding if the file exists"""
        if self.reference_se is not None:
            return self.reference_se

        ref_path = os.getenv(
            "REF_VOICE_PATH", "assets/voices/reference_audio.wav"
            )
        if not os.path.exists(ref_path):
            self.logger.warning(
                f"Reference voice file {ref_path} not found. "
                "Creating a placeholder."
            )
            os.makedirs('assets/voices', exist_ok=True)
            placeholder_path = 'assets/voices/reference_audio.wav'
            torchaudio.save(
                placeholder_path,
                torch.zeros((1, 16000)),  # Ensure correct shape
                16000
            )
            ref_path = placeholder_path

        try:
            self.reference_se = se_extractor.get_se(
                ref_path,
                tone_color_converter=self.tone_color_converter
            )
            return self.reference_se
        except Exception as e:
            self.logger.error(f"Failed to load reference voice: {e}")
            return None

    async def get_stt_model(self):
        """Lazy load Whisper model"""
        if self._stt_model is None:
            self.logger.info("Loading STT model...")
            self._stt_model = WhisperModel(
                "large",
                device=self.device,
                compute_type=self.compute_type
            )
            self.logger.info("STT model loaded")
        return self._stt_model

    async def get_tts_model(self):
        """Lazy load TTS model"""
        if self._tts_model is None:
            self.logger.info("Loading TTS model...")
            self._tts_model = BaseSpeakerTTS(device=self.device)
            self.logger.info("TTS model loaded")
        return self._tts_model

    async def cleanup_resources(self):
        """Release resources to free memory"""
        if self._stt_model is not None:
            del self._stt_model
            self._stt_model = None
        if self._tts_model is not None:
            del self._tts_model
            self._tts_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Audio processor resources cleaned up")

    def detect_background_noise(self, audio: np.ndarray) -> dict:
        """
        Detects background noise level from an audio signal.

        Args:
            audio (np.ndarray): The audio waveform data

        Returns:
            dict: Noise level metrics
        """
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return {"error": "Invalid or empty audio data"}

        # Compute basic noise metrics
        noise_level = np.std(audio)  # Standard deviation as a noise estimate
        mean_amplitude = np.mean(np.abs(audio))  # Mean absolute amplitude
        peak_amplitude = np.max(np.abs(audio))  # Peak amplitude

        return {
            "noise_level": noise_level,
            "mean_amplitude": mean_amplitude,
            "peak_amplitude": peak_amplitude,
            "is_noisy": noise_level > 0.02  # threshold for "noisy"
        }

    async def record_audio_async(self, duration: float = 5) -> np.ndarray:
        """
        Record audio asynchronously

        Args:
            duration: Recording duration in seconds

        Returns:
            Recorded audio as numpy array
        """
        try:
            sample_rate = 16000
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            await asyncio.sleep(duration)
            sd.wait()
            return audio.flatten()
        except Exception as e:
            self.logger.error(f"Recording failed: {e}")
            return np.zeros(int(duration * sample_rate))

    async def transcribe(
            self, audio: np.ndarray, sample_rate: int = 24000
            ) -> Dict:
        """
        Transcribe audio to text with word timestamps

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Dict with transcription text and metadata
        """
        try:
            # Ensure audio is a NumPy array and has enough length
            if not isinstance(audio, np.ndarray) or len(audio) < 160:
                raise ValueError(
                    "Input audio too short or invalid for analysis"
                    )

            # Resample to 16kHz for Whisper if needed
            if sample_rate != 16000:
                audio_tensor = torch.from_numpy(audio)
                audio_resampled = torchaudio.functional.resample(
                    audio_tensor, sample_rate, 16000
                ).numpy()
            else:
                audio_resampled = audio

            # Store in buffer for prosody analysis
            self.audio_buffer = audio_resampled

            # Get STT model and transcribe with word timestamps
            model = await self.get_stt_model()
            segments, info = model.transcribe(
                audio_resampled,
                word_timestamps=True,
                language="en"  # Default language set to English
            )

            # Process results
            transcription = " ".join(segment.text for segment in segments)
            words = [
                {"word": w.word, "start": w.start, "end": w.end}
                for segment in segments for w in segment.words
            ]

            # Analyze prosody if we have segments
            prosody = self._analyze_prosody(segments) if segments else {
                'pitch_range': 0, 'speaking_rate': 0
            }

            return {
                "text": transcription,
                "language": info.language if info else "en",
                "words": words,
                "prosody": prosody,
                "duration": (
                    info.duration if info else len(audio_resampled) / 16000
                )
            }

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return {"text": "", "error": str(e)}

    def _analyze_prosody(self, segments) -> Dict[str, float]:
        """Simplified prosody analysis."""
        if not segments:
            return {'pitch_range': 0, 'speaking_rate': 0}
        return {'pitch_range': 0.5, 'speaking_rate': 1.0}  # Placeholder values

    async def synthesize(
            self, text: str, emotion: str = "neutral",
            intensity: float = 1.0, voice_params: Optional[dict] = None
    ) -> np.ndarray:
        """
        Synthesize speech with emotion and voice customization

        Args:
            text: Text to synthesize
            emotion: Emotion style ("happy", "sad", "angry", "neutral")
            intensity: Intensity factor (0.5-2.0)
            voice_params: Optional voice customization parameters

        Returns:
            Synthesized audio as numpy array
        """
        if not text:
            self.logger.warning("Empty text provided for synthesis")
            return np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)

        valid_emotions = {"happy", "sad", "angry", "neutral"}
        if emotion.lower() not in valid_emotions:
            self.logger.warning(
                f"Invalid emotion: {emotion}, using neutral instead"
                )
            emotion = "neutral"

        # Ensure reference voice is loaded
        if self.reference_se is None:
            self._load_reference_voice()

        # Get TTS model
        tts_model = await self.get_tts_model()
        src_path, target_path = None, None

        try:
            # Generate base speech audio
            src_path = await asyncio.to_thread(
                tts_model.generate_audio,
                text,
                speaker='emotional',
                style=emotion,
                speed=max(0.5, min(2.0, intensity))  # Clamp speed 0.5x and 2x
            )

            target_path = "temp_output.wav"

            # Apply tone color conversion
            target_se = (
                voice_params.get("tone_color")
                if voice_params else self.reference_se
            )
            self.tone_color_converter.convert(
                src_path=src_path,
                target_se=(
                    target_se or self.reference_se
                ),  # Fallback to reference if not provided
                output_path=target_path
            )

            # Load and return the processed audio
            audio, sr = torchaudio.load(target_path)
            if audio.numel() == 0:
                raise ValueError("Generated audio is empty")

            # Ensure correct sampling rate
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, sr, self.sample_rate
                    )

            return audio.numpy().flatten()

        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            return np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)
        # Fallback silence

        finally:
            # Cleanup temporary files
            for path in [src_path, target_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    def play_audio(self, audio: np.ndarray, sample_rate: int = 24000):
        """Simplified audio playback."""
        try:
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            self.logger.error(f"Audio playback failed: {e}")

    async def speak_text(self, text: str, emotion: str = "neutral", intensity: float = 1.0, voice_params: Optional[dict] = None):
        """
        High-level wrapper: text → waveform → playback.
        Synthesizes speech from text and plays it back.
        Args:
            text (str): Text to speak.
            emotion (str): Emotion style ("happy", "sad", "angry", "neutral").
            intensity (float): Intensity factor (0.5-2.0).
            voice_params (dict, optional): Voice customization.
        """
        try:
            audio = await self.synthesize(text, emotion, intensity, voice_params)
            if audio is not None and audio.size > 0:
                self.play_audio(audio, sample_rate=self.sample_rate)
            else:
                self.logger.warning("No audio generated for speak_text.")
        except Exception as e:
            self.logger.error(f"speak_text failed: {e}")

