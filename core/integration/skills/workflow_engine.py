from utils.config_loader import get_config, cached_config
import pkgutil
import importlib
from typing import Dict, Callable, Any
from core.integration.skills.api_integration import APIManager
from core.integration.skills.validation_utils import validate_parameters, retry_async_call
import logging
import json

logger = logging.getLogger(__name__)


class WorkflowEngine:
    def __init__(self):
        # Load workflows definitions
        self.workflows = self._load_workflows()
        # Initialize API manager and skill registry
        self.api_manager = APIManager()
        self.skill_handlers: Dict[str, Callable[..., Any]] = {}
        # Dynamically discover and register skill modules
        self._register_skills()

    def _load_workflows(self) -> Dict:
        return get_config("workflows.yaml") or {}

    def _register_skills(self):
        """Dynamically load, register skills under core.integration.skills."""
        import core.integration.skills as skills_pkg
        for finder, name, ispkg in pkgutil.iter_modules(skills_pkg.__path__):
            module = importlib.import_module(f"core.integration.skills.{name}")
            # Find the first class in module with an 'execute' method
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and callable(getattr(attr, "execute", None)):
                    instance = attr()
                    # Use SERVICE_NAME if defined, else module name
                    service_name = getattr(module, "SERVICE_NAME", name)
                    self.skill_handlers[service_name] = instance.execute
                    break

    async def execute_workflow(self, workflow_id: str, inputs: Dict) -> Dict:
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return {"error": f"Workflow '{workflow_id}' not found", "status": "error"}
            
        results = {}
        errors = []
        
        for step_index, step in enumerate(workflow.get("steps", [])):
            name = step.get("name", f"step_{step_index}")
            step_type = step.get("type")
            action = step.get("action")
            
            # Check if previous steps have critical errors
            if any(err.get("critical", False) for err in errors):
                results[name] = {
                    "status": "skipped", 
                    "error": "Skipped due to previous critical error"
                }
                continue
                
            try:
                # Resolve parameters using inputs and previous step results
                params = self._resolve_parameters(
                    step.get("parameters", {}), inputs, results
                )
                
                # --- Schema validation ---
                schema = step.get("schema")
                if schema:
                    try:
                        await validate_parameters(params, schema)
                    except Exception as e:
                        error_msg = f"Parameter validation failed: {e}"
                        logger.error(f"Step '{name}' {error_msg}")
                        results[name] = {"status": "error", "error": error_msg}
                        errors.append({"step": name, "error": error_msg, "critical": True})
                        continue
                
                # --- Skill/API invocation with retry/timeout ---
                if step_type in ("api_call", "skill"):
                    service, _, method = action.partition('.')
                    handler = self.skill_handlers.get(service)
                    call_params = params.copy()
                    if method:
                        call_params["action"] = method
                        
                    try:
                        if handler:
                            # Skill handler invocation
                            step_result = await retry_async_call(handler, call_params)
                        else:
                            # Fallback to generic APIManager
                            step_result = await retry_async_call(
                                self.api_manager.execute, service, call_params
                            )
                            
                        # Check for errors in the result
                        if isinstance(step_result, dict) and "error" in step_result:
                            error_msg = step_result["error"]
                            is_critical = step.get("critical", False)
                            errors.append({"step": name, "error": error_msg, "critical": is_critical})
                            step_result["status"] = "error"
                        else:
                            step_result["status"] = "success"
                            
                        results[name] = step_result
                        
                    except Exception as e:
                        error_msg = f"Step execution failed: {e}"
                        logger.error(f"Step '{name}' {error_msg}")
                        results[name] = {"status": "error", "error": error_msg}
                        errors.append({"step": name, "error": error_msg, "critical": step.get("critical", False)})
                        
                elif step_type == "data_processing":
                    try:
                        process_result = self._process_data(action, params)
                        if process_result is None:
                            error_msg = f"Data processing returned None for operation '{action}'"
                            results[name] = {"status": "error", "error": error_msg}
                            errors.append({"step": name, "error": error_msg, "critical": False})
                        else:
                            results[name] = {"status": "success", "data": process_result}
                    except Exception as e:
                        error_msg = f"Data processing failed: {e}"
                        logger.error(f"Data processing step '{name}' {error_msg}")
                        results[name] = {"status": "error", "error": error_msg}
                        errors.append({"step": name, "error": error_msg, "critical": False})
                else:
                    error_msg = f"Unsupported step type: {step_type}"
                    results[name] = {"status": "error", "error": error_msg}
                    errors.append({"step": name, "error": error_msg, "critical": False})
                    
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                logger.error(f"Workflow step '{name}' failed with {error_msg}")
                results[name] = {"status": "error", "error": error_msg}
                errors.append({"step": name, "error": error_msg, "critical": False})
        
        # Create the final output
        output = self._format_output(workflow.get("output", {}), results)
        
        # Add workflow execution summary
        output["workflow_id"] = workflow_id
        output["status"] = "error" if errors else "success"
        if errors:
            output["errors"] = errors
            
        return output

    def _resolve_parameters(
            self, params_template: Dict, inputs: Dict, results: Dict
            ) -> Dict:
        params = {}
        for key, value in params_template.items():
            if isinstance(value, str) and value.startswith("{inputs."):
                params[key] = inputs.get(value[8:-1], "")
            elif isinstance(value, str) and value.startswith("{results."):
                params[key] = results.get(value[9:-1], "")
            else:
                params[key] = value
        return params

    def _process_data(self, operation: str, data: Dict):
        if operation == "merge":
            return {**data.get("source1", {}), **data.get("source2", {})}
        if operation == "filter":
            return [
                item for item in data.get("items", [])
                if item.get(data.get("key")) == data.get("value")
            ]
        return None

    def _format_output(self, output_template: Dict, results: Dict) -> Dict:
        return {key: results.get(val) for key, val in output_template.items()}
