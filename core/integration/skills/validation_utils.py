import jsonschema
import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def validate_parameters(params: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate parameters against a JSON schema."""
    try:
        jsonschema.validate(instance=params, schema=schema)
    except jsonschema.ValidationError as e:
        logger.error(f"Parameter validation failed: {e.message}")
        raise

async def retry_async_call(callable_fn, *args, retries=3, timeout=10, **kwargs):
    """Retry an async callable with timeout and exponential backoff."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return await asyncio.wait_for(callable_fn(*args, **kwargs), timeout=timeout)
        except Exception as e:
            last_exc = e
            logger.warning(f"Attempt {attempt} failed: {e}")
            await asyncio.sleep(min(2 ** attempt, 10))
    logger.error(f"All {retries} attempts failed.")
    raise last_exc
