import logging
import functools
from utils.model_switcher import SecurityError
from enum import Enum
from typing import Callable, Any, Coroutine
import sentry_sdk  # Assuming Sentry is used for error tracking


class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""
    CRITICAL = "CRITICAL"
    MODERATE = "MODERATE"


class ErrorHandler:
    """
    A decorator-based error handling class that provides logging and
    optional error tracking.
    
    Supports two severity levels:
    - CRITICAL: Raises a SystemError after logging
    - MODERATE: Logs the error and returns None
    """
    
    @classmethod
    def configure_logger(cls, logger_name: str = __name__,
                         log_level: int = logging.DEBUG,
                         log_format: str = '%(asctime)s - %(name)s - '
                                           '%(levelname)s - %(message)s'):
        """
        Configures and returns a logger with specified parameters.
        
        :param logger_name: Name of the logger
        :param log_level: Logging level (default: DEBUG)
        :param log_format: Format of log messages
        :return: Configured logger
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        
        # Clear any existing handlers to prevent duplicate logging
        logger.handlers.clear()
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    # Class-level logger with default configuration
    logger = configure_logger.__func__()
    
    @classmethod
    def handle_error(cls, component: str,
                     severity: ErrorSeverity = ErrorSeverity.MODERATE
                     ) -> Callable:
        """
        Decorator for handling errors with specified severity.
        
        :param component: Name of the component/module where error occurs
        :param severity: Error severity level (default: MODERATE)
        :return: Decorator function
        """
        def decorator(func: Callable[..., Coroutine[Any, Any, Any]]
                      ) -> Callable[..., Coroutine[Any, Any, Any]]:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    cls._log_error(component, e, severity)
                    
                    if severity == ErrorSeverity.CRITICAL:
                        raise SystemError(
                            f"Critical failure in {component}"
                        ) from e
                    
                    return None
            return wrapper
        return decorator
    
    @classmethod
    def _log_error(cls, component: str, error: Exception, 
                   severity: ErrorSeverity):
        if isinstance(error, SecurityError):
            severity = ErrorSeverity.CRITICAL
        """
        Log the error and optionally capture it in Sentry.
        
        :param component: Name of the component/module
        :param error: Exception that occurred
        :param severity: Error severity level
        """
        error_message = f"[{severity.value}] {component} failure: {str(error)}"
        cls.logger.error(error_message)
        
        # Optional Sentry error tracking
        try:
            sentry_sdk.capture_exception(error)
        except Exception as sentry_error:
            cls.logger.warning(f"Failed to capture error in Sentry: "
                               f"{sentry_error}")


# Example usage
class ExampleService:
    @ErrorHandler.handle_error(component="user_authentication",
                               severity=ErrorSeverity.CRITICAL)
    async def authenticate_user(self, username: str, password: str) -> bool:
        # Simulated authentication logic
        if not username or not password:
            raise ValueError("Invalid credentials")
        return True


# Optional: Additional setup for Sentry (if used)
def initialize_sentry(dsn: str):
    """
    Initialize Sentry SDK with the provided DSN.
    
    :param dsn: Sentry Data Source Name
    """
    sentry_sdk.init(dsn=dsn)