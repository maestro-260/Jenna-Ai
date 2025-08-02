import unittest
import os
import sys
import asyncio
from pathlib import Path
import tempfile
import shutil
import logging

# Add project root to path for imports
root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)

# Disable verbose logging during tests
logging.basicConfig(level=logging.ERROR)

class TestJennaSystemStartup(unittest.TestCase):
    """System test to verify JENNA system can start up properly."""
    
    def setUp(self):
        """Set up test environment with temporary directories."""
        # Create a temporary test directory
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.test_dir, "config")
        self.models_dir = os.path.join(self.test_dir, "models")
        self.memory_dir = os.path.join(self.test_dir, "memory")
        self.logs_dir = os.path.join(self.test_dir, "logs")
        
        # Create required directories
        for directory in [self.config_dir, self.models_dir, self.memory_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Store original environment
        self.original_env = {
            "JENNA_CONFIG_DIR": os.environ.get("JENNA_CONFIG_DIR"),
            "LOG_LEVEL": os.environ.get("LOG_LEVEL")
        }
        
        # Set test environment
        os.environ["JENNA_CONFIG_DIR"] = self.config_dir
        os.environ["LOG_LEVEL"] = "ERROR"  # Minimize noise during testing
        
        # Copy necessary config files from project to test dir
        project_config_dir = os.path.join(root_path, "config")
        if os.path.exists(project_config_dir):
            for config_file in os.listdir(project_config_dir):
                src = os.path.join(project_config_dir, config_file)
                dst = os.path.join(self.config_dir, config_file)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
        
        # Restore original environment
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    async def async_test_system_startup(self):
        """Test that JENNA system can initialize correctly."""
        from services.main_service import JENNA
        
        # Create JENNA instance
        jenna = JENNA()
        
        try:
            # Initialize JENNA with a timeout
            initialization_task = asyncio.create_task(jenna.initialize())
            
            # Add a timeout to prevent test from hanging indefinitely
            try:
                await asyncio.wait_for(initialization_task, timeout=30.0)
                
                # Check that initialization was successful
                self.assertTrue(jenna.initialized, "JENNA initialization failed")
                
                # Check core components were initialized
                self.assertIsNotNone(jenna.reasoner, "Reasoner component not initialized")
                self.assertIsNotNone(jenna.audio, "Audio component not initialized")
                self.assertIsNotNone(jenna.context_db, "Context DB not initialized")
                
            except asyncio.TimeoutError:
                self.fail("JENNA initialization timed out after 30 seconds")
                
        finally:
            # Clean up resources
            if hasattr(jenna, '_cleanup_resources'):
                await jenna._cleanup_resources()
    
    def test_system_startup(self):
        """Run the async test with event loop."""
        asyncio.run(self.async_test_system_startup())
    
    def test_bootstrap_import(self):
        """Test that bootstrap module can be imported."""
        try:
            import bootstrap
            self.assertTrue(callable(bootstrap.setup_environment))
        except ImportError as e:
            self.fail(f"Failed to import bootstrap module: {e}")
    
    def test_main_service_import(self):
        """Test that main_service module can be imported."""
        try:
            from services.main_service import JENNA
            # Create instance to verify class definition is valid
            jenna = JENNA()
            self.assertTrue(hasattr(jenna, 'run'))
            self.assertTrue(callable(jenna.run))
        except ImportError as e:
            self.fail(f"Failed to import main_service module: {e}")
        except Exception as e:
            self.fail(f"Error creating JENNA instance: {e}")
            
if __name__ == "__main__":
    unittest.main() 