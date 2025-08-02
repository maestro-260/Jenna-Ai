import unittest
import os
import tempfile
from pathlib import Path
import yaml
import shutil
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.config_loader import get_config, reload_config, _resolve_env_vars

class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test configs
        self.test_dir = tempfile.mkdtemp()
        self.original_config_dir = os.environ.get("JENNA_CONFIG_DIR")
        os.environ["JENNA_CONFIG_DIR"] = self.test_dir
        
        # Create sample configs for testing
        self.sample_base_config = {
            "model": {
                "name": "base_model",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            },
            "api_key": "${API_KEY}",
            "debug": True
        }
        
        self.sample_dev_config = {
            "model": {
                "parameters": {
                    "temperature": 0.9  # Override base temperature
                }
            },
            "debug": True
        }
        
        self.sample_prod_config = {
            "model": {
                "parameters": {
                    "temperature": 0.5  # Override base temperature
                }
            },
            "debug": False
        }
        
        # Write sample configs to temp directory
        with open(os.path.join(self.test_dir, "test_config.yaml"), "w") as f:
            yaml.dump(self.sample_base_config, f)
            
        with open(os.path.join(self.test_dir, "test_config.dev.yaml"), "w") as f:
            yaml.dump(self.sample_dev_config, f)
            
        with open(os.path.join(self.test_dir, "test_config.prod.yaml"), "w") as f:
            yaml.dump(self.sample_prod_config, f)
            
        # Set test environment variable
        os.environ["API_KEY"] = "test_api_key_12345"
    
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        
        # Restore original config dir
        if self.original_config_dir:
            os.environ["JENNA_CONFIG_DIR"] = self.original_config_dir
        else:
            os.environ.pop("JENNA_CONFIG_DIR", None)
            
        # Remove test env var
        os.environ.pop("API_KEY", None)
    
    def test_basic_config_loading(self):
        """Test basic config loading works"""
        config = get_config("test_config.yaml", validate=False)
        self.assertEqual(config["model"]["name"], "base_model")
        self.assertEqual(config["model"]["parameters"]["temperature"], 0.7)
        self.assertEqual(config["debug"], True)
    
    def test_env_specific_config(self):
        """Test environment-specific config loading works"""
        # Load dev config
        dev_config = get_config("test_config.yaml", validate=False, env="dev")
        self.assertEqual(dev_config["model"]["name"], "base_model")  # Inherited from base
        self.assertEqual(dev_config["model"]["parameters"]["temperature"], 0.9)  # Overridden
        
        # Load prod config
        prod_config = get_config("test_config.yaml", validate=False, env="prod")
        self.assertEqual(prod_config["model"]["name"], "base_model")  # Inherited from base
        self.assertEqual(prod_config["model"]["parameters"]["temperature"], 0.5)  # Overridden
        self.assertEqual(prod_config["debug"], False)  # Overridden
    
    def test_env_var_resolution(self):
        """Test environment variable resolution works"""
        config = get_config("test_config.yaml", validate=False)
        self.assertEqual(config["api_key"], "test_api_key_12345")
        
        # Test the resolver directly
        test_data = {
            "simple": "${API_KEY}",
            "nested": {"key": "${API_KEY}"},
            "list": ["${API_KEY}", "other"]
        }
        resolved = _resolve_env_vars(test_data)
        self.assertEqual(resolved["simple"], "test_api_key_12345")
        self.assertEqual(resolved["nested"]["key"], "test_api_key_12345")
        self.assertEqual(resolved["list"][0], "test_api_key_12345")
    
    def test_config_reload(self):
        """Test config reloading works"""
        # Load initial config
        config1 = get_config("test_config.yaml", validate=False)
        
        # Change the config file
        modified_config = self.sample_base_config.copy()
        modified_config["model"]["name"] = "modified_model"
        with open(os.path.join(self.test_dir, "test_config.yaml"), "w") as f:
            yaml.dump(modified_config, f)
        
        # Without reload, should get cached config
        config2 = get_config("test_config.yaml", validate=False)
        self.assertEqual(config2["model"]["name"], "base_model")  # Still the old value
        
        # After reload, should get new config
        reload_config("test_config.yaml")
        config3 = get_config("test_config.yaml", validate=False)
        self.assertEqual(config3["model"]["name"], "modified_model")  # Updated value

if __name__ == "__main__":
    unittest.main() 