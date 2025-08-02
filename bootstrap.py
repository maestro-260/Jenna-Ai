import sys
import os
import logging
from pathlib import Path
import argparse
import asyncio
import importlib.util

logger = logging.getLogger(__name__)

def setup_environment(env="dev"):
    """
    Set up the environment for JENNA to run, including configuration
    paths, environment variables, and logging.
    """
    # Set config dir env if not set
    config_dir = os.environ.get("JENNA_CONFIG_DIR")
    if not config_dir:
        default = Path(__file__).parent / "config"
        os.environ["JENNA_CONFIG_DIR"] = str(default.resolve())
        
    # Set environment-specific configurations
    if env == "dev":
        os.environ["LOG_LEVEL"] = os.environ.get("LOG_LEVEL", "DEBUG")
        os.environ["OLLAMA_HOST"] = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    elif env == "prod":
        os.environ["LOG_LEVEL"] = os.environ.get("LOG_LEVEL", "INFO")
        
    # Set up logging
    log_level = getattr(logging, os.environ["LOG_LEVEL"])
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/jenna.log")
        ]
    )
    
    # Create required directories
    for directory in ["logs", "memory", "models", "assets/voices"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Environment set up for: {env}")
    logger.info(f"Config directory: {os.environ['JENNA_CONFIG_DIR']}")
    return True

def check_dependencies():
    """Check critical dependencies before starting"""
    required_modules = ["torch", "sounddevice", "yaml", "vosk", "ollama"]
    missing = []
    
    for module in required_modules:
        if not importlib.util.find_spec(module):
            missing.append(module)
    
    if missing:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        logger.error("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    
    return True

async def run_jenna(args):
    """Run the JENNA AI system"""
    from services.main_service import JENNA
    
    jenna = JENNA()
    try:
        # Run until interrupted
        await jenna.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested via keyboard interrupt")
    except Exception as e:
        logger.error(f"Error running JENNA: {e}")
    finally:
        # JENNA's run method should handle cleanup of resources
        logger.info("JENNA has been shut down")

def main():
    """Main entry point for the JENNA system"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="JENNA AI System")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev", 
                        help="Environment to run in (dev/prod)")
    parser.add_argument("--config", type=str, 
                        help="Path to custom config directory")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode with extra logging")
    parser.add_argument("--no-run", action="store_true", 
                        help="Setup environment but don't start JENNA")
    args = parser.parse_args()
    
    # Set custom config path if provided
    if args.config:
        os.environ["JENNA_CONFIG_DIR"] = str(Path(args.config).resolve())
    
    # Set debug mode if requested
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
    
    # Set up the environment
    if not setup_environment(args.env):
        logger.error("Environment setup failed")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Run JENNA unless --no-run is specified
    if not args.no_run:
        asyncio.run(run_jenna(args))
    else:
        logger.info("Environment setup complete. Not starting JENNA (--no-run specified)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
