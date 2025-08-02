import os
from pathlib import Path
import yaml
import json
import re
from typing import Any, Dict, Optional

try:
    from cerberus import Validator
    _HAS_CERBERUS = True
except ImportError:
    _HAS_CERBERUS = False

try:
    from pydantic import BaseModel, create_model
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False

CONFIG_DIR = Path(os.environ.get("JENNA_CONFIG_DIR", Path(__file__).parent.parent / "config"))

# Example schemas for validation (expand as needed)
SCHEMAS = {
    "model.yaml": {
        "model_name": {'type': 'string', 'required': True},
        "parameters": {'type': 'dict', 'required': False},
    },
    "constraints.yaml": {
        "constraints": {'type': 'list', 'required': True},
    },
    # Add schemas for other YAMLs as needed
}

# Caching configs for performance
_configs_cache = {}

def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve environment variables in config values."""
    if isinstance(value, str):
        # Match ${ENV_VAR} or $ENV_VAR patterns
        pattern = r'\${([A-Za-z0-9_]+)}|\$([A-Za-z0-9_]+)'
        
        def replace_env_var(match):
            env_var = match.group(1) or match.group(2)
            return os.environ.get(env_var, f"${env_var}")
            
        return re.sub(pattern, replace_env_var, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value

def load_yaml(filename: str, validate: bool = True, env: str = None) -> Dict[str, Any]:
    """Load YAML config with optional schema validation.
    
    Args:
        filename: Name of the YAML file to load
        validate: Whether to validate against schema
        env: Optional environment (dev/prod) to load environment-specific configs
        
    Returns:
        Loaded and processed configuration dictionary
    """
    # Determine the base path
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    # Load the base configuration
    with open(path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # Load environment-specific overrides if specified
    final_config = base_config.copy()
    if env:
        env_file = path.with_stem(f"{path.stem}.{env}")
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                env_config = yaml.safe_load(f)
                # Deep merge the configurations
                final_config = _deep_merge(final_config, env_config)
    
    # Load secrets from environment variables
    final_config = _resolve_env_vars(final_config)
    
    # Validate if requested and possible
    if validate:
        if _HAS_PYDANTIC and filename.endswith((".yaml", ".yml")):
            # Try to create a model from the schema and validate
            schema_file = path.with_suffix(".schema.json")
            if schema_file.exists():
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                model_fields = _convert_schema_to_pydantic(schema)
                model = create_model(f"{filename.split('.')[0].capitalize()}Config", **model_fields)
                # Validate the config against the model
                try:
                    model(**final_config)
                except Exception as e:
                    raise ValueError(f"Validation failed for {filename}: {e}")
        elif _HAS_CERBERUS:
            # Use Cerberus for validation as fallback
            schema_path = path.with_stem(f"{path.stem}.schema")
            if schema_path.exists():
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema = yaml.safe_load(f)
                v = Validator(schema)
                if not v.validate(final_config):
                    raise ValueError(f"Schema validation failed for {filename}: {v.errors}")
    
    return final_config

def get_config(filename: str, validate: bool = True, env: str = None) -> Dict[str, Any]:
    """Public interface for config access with environment support."""
    cache_key = f"{filename}_{env or 'default'}"
    if cache_key not in _configs_cache:
        _configs_cache[cache_key] = load_yaml(filename, validate, env)
    return _configs_cache[cache_key]

def cached_config(filename: str, validate: bool = True, env: str = None):
    """Legacy alias for get_config for backward compatibility."""
    return get_config(filename, validate, env)

def reload_config(filename: str = None):
    """Force reload of configs from disk, optionally for a specific file."""
    global _configs_cache
    if filename:
        # Clear specific config cache entries
        keys_to_remove = [k for k in _configs_cache if k.startswith(f"{filename}_")]
        for key in keys_to_remove:
            del _configs_cache[key]
    else:
        # Clear entire cache
        _configs_cache = {}

def _deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """Recursively merge two dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def _convert_schema_to_pydantic(schema: Dict) -> Dict:
    """Convert JSON Schema to Pydantic field definitions (simplified)."""
    # This is a very basic implementation - would need to be expanded
    # for a production-ready solution
    result = {}
    properties = schema.get("properties", {})
    
    for prop_name, prop_schema in properties.items():
        required = prop_name in schema.get("required", [])
        field_type = prop_schema.get("type", "any")
        
        # Map JSON Schema types to Python types (simplified)
        if field_type == "string":
            python_type = str
        elif field_type == "integer":
            python_type = int
        elif field_type == "number": 
            python_type = float
        elif field_type == "boolean":
            python_type = bool
        elif field_type == "array":
            python_type = list
        elif field_type == "object":
            python_type = dict
        else:
            python_type = Any
            
        # Handle required vs optional fields
        if required:
            result[prop_name] = (python_type, ...)
        else:
            default = prop_schema.get("default", None)
            result[prop_name] = (Optional[python_type], default)
    
    return result
