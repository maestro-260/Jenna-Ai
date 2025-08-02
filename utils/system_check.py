# Add to utils/system_check.py
import sounddevice as sd
import sys
import os
import logging

logger = logging.getLogger(__name__)


async def verify_environment():
    """Verify system requirements"""
    try:
        # Check Python version
        if sys.version_info < (3, 11, 0):
            raise RuntimeError("Python 3.11.0+ required")

        # Check directories
        required_dirs = ["memory", "assets/voices", "models", "config", "logs"]
        for d in required_dirs:
            if not os.path.exists(d):
                os.makedirs(d)

        # Check audio device
        if not sd.query_devices():
            raise RuntimeError("No audio devices found")

    except Exception as e:
        logger.error(f"Environment check failed: {e}")
        raise
