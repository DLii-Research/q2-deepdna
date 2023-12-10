from collections import namedtuple
import inspect
from typing import Dict, List, Optional

# A field definition for a Qiime2 action method.
Field = namedtuple("MethodField", ["semantic_type", "description"])

def register_format(cls):
    """
    Register a format with the plugin.
    """
    from .plugin_setup import plugin
    plugin.register_formats(cls)
    return cls


def register_method(
        name: str,
        *,
        description: Optional[str] = None,
        inputs: Optional[Dict[str, Field]] = None,
        parameters: Optional[Dict[str, Field]] = None,
        outputs: Optional[Dict[str, Field]] = None,
        citations: Optional[List] = None,
):
    """
    Register a method with the plugin.
    """
    inputs = inputs if inputs is not None else {}
    parameters = parameters if parameters is not None else {}
    outputs = outputs if outputs is not None else {}
    citations = citations if citations is not None else []
    def decorator(func):
        # Assert that the function defines all of the inputs, parameters, and outputs
        signature = inspect.signature(func)
        expected_arguments = set(inputs.keys()) | set(parameters.keys())
        actual_arguments = set(signature.parameters.keys())
        if expected_arguments != actual_arguments:
            raise Exception(f"Missing arguments for {name}: {expected_arguments - actual_arguments}")
        # Register the plugin
        from .plugin_setup import plugin
        plugin.methods.register_function(
            function=func,
            inputs={k: v.semantic_type for k, v in inputs.items()},
            parameters={k: v.semantic_type for k, v in parameters.items()},
            outputs={k: v.semantic_type for k, v in outputs.items()},
            input_descriptions={k: v.description for k, v in inputs.items()},
            parameter_descriptions={k: v.description for k, v in parameters.items()},
            output_descriptions={k: v.description for k, v in outputs.items()},
            name=name,
            description=description,
            citations=citations,
        )
        return func
    return decorator
