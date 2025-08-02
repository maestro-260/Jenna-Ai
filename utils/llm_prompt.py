import asyncio
import os
import logging

try:
    import ollama
except ImportError:
    ollama = None

logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

async def dynamic_llm_prompt(prompt: str, model: str = None) -> str:
    """
    Query Ollama or another local LLM for a dynamic response to a prompt.
    Falls back to a default message if no LLM is available.
    """
    model = model or OLLAMA_MODEL
    if ollama:
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            # Ollama returns a dict with 'message' key
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Ollama LLM call failed: {e}")
            return "(I'm ready for your command.)"
    else:
        logger.warning("Ollama is not installed. Returning fallback message.")
        return "(I'm ready for your command.)"
