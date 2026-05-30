# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import importlib
import inspect
import sys
import types
import warnings
from dataclasses import fields
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

import pytest

from dvlt.config import schema
from dvlt.config.schema import register_configs


# Check Python version
PY_310_PLUS = sys.version_info >= (3, 10)

# Map of classes to parameters that are intentionally omitted from their configs
# because they're passed during instantiation
INJECTED_PARAMS = {
    "dvlt.engine.trainer.Trainer": ["model", "data"],
}


def get_config_classes():
    """Get all config classes from schema.py that have a _target_ attribute."""
    result = []
    for name in dir(schema):
        attr = getattr(schema, name)
        if (
            hasattr(attr, "__dataclass_fields__")
            and "_target_" in attr.__dataclass_fields__
            and name != "ModelConfig"  # Skip base class
        ):
            result.append(attr)

    return result


def get_constructor_params(cls):
    """Get the constructor parameters and their default values for a class."""
    signature = inspect.signature(cls.__init__)
    params = {}

    for name, param in signature.parameters.items():
        if name not in ["self", "args", "kwargs"]:
            params[name] = param.default if param.default is not inspect.Parameter.empty else None

    return params


def import_target_class(target_path):
    """Import a class from its path string."""
    module_path, class_name = target_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def is_hydra_dict_type(type_hint):
    """Check if a type is Dict[str, Any], which is a common Hydra instantiation pattern."""
    if get_origin(type_hint) is not dict:
        return False
    args = get_args(type_hint)
    return len(args) == 2 and args[0] is str and args[1] is Any


def is_optional_type(type_hint):
    """Check if a type is Optional[T] or Union[T, None] across Python versions."""
    origin = get_origin(type_hint)

    # In Python 3.8-3.9, Optional[T] has an origin of typing.Union
    # In Python 3.10+, it can be types.UnionType (from the | operator)
    if origin in (Union, types.UnionType if PY_310_PLUS else None):
        args = get_args(type_hint)
        return len(args) == 2 and type(None) in args

    return False


def unwrap_optional(type_hint):
    """Extract the inner type from Optional[T] or Union[T, None]."""
    args = get_args(type_hint)
    return args[0] if args[1] is type(None) else args[1]


def describe_type(type_obj):
    """Return a descriptive string for a type object, for better error messages."""
    if get_origin(type_obj) is None:
        return str(type_obj)

    origin = get_origin(type_obj)
    args = get_args(type_obj)

    if origin is list:
        return f"List[{describe_type(args[0])}]"
    elif origin is dict:
        return f"Dict[{describe_type(args[0])}, {describe_type(args[1])}]"
    elif origin is Optional:
        return f"Optional[{describe_type(args[0])}]"
    elif origin in (Union, types.UnionType if PY_310_PLUS else None):
        # Check if it's actually an Optional type (Union with NoneType)
        if is_optional_type(type_obj):
            non_none_type = unwrap_optional(type_obj)
            return f"Optional[{describe_type(non_none_type)}]"
        else:
            return f"Union[{', '.join(describe_type(arg) for arg in args)}]"
    else:
        return f"{origin}[{', '.join(describe_type(arg) for arg in args)}]"


def is_compatible_type(config_type, target_type):
    """Check if a config field type is compatible with a target parameter type.

    Accounts for Hydra instantiation patterns where Dict[str, Any] or List[Dict[str, Any]]
    in configs can map to concrete types in implementation.
    """
    # Handle simple cases
    if config_type == target_type:
        return True

    # Special case: Any type in config can match any target type
    # This allows flexible fields that can be int/float or tuple/list
    if config_type is Any:
        return True

    # Special case: str in config can reference a Callable in implementation
    # This is a common Hydra pattern for function references
    if config_type is str and get_origin(target_type) is collections.abc.Callable:
        return True

    # Check if target is Optional/Union
    if is_optional_type(target_type):
        # For Optional types, check if config type is compatible with the inner type
        non_none_type = unwrap_optional(target_type)

        # Special case: List[Dict[str, Any]] in config and Optional[List[SomeClass]] in target
        if (
            get_origin(config_type) is list
            and get_origin(non_none_type) is list
            and len(get_args(config_type)) == 1
            and len(get_args(non_none_type)) == 1
        ):
            if is_hydra_dict_type(get_args(config_type)[0]):
                return True

        # Otherwise, check if config type is compatible with non-None type
        return is_compatible_type(config_type, non_none_type)

    # Handle Union types that aren't Optional
    union_origins = [Union]
    if PY_310_PLUS:
        union_origins.append(types.UnionType)

    if get_origin(target_type) in union_origins and not is_optional_type(target_type):
        # If any member of the union is compatible, consider it compatible
        return any(is_compatible_type(config_type, arg) for arg in get_args(target_type))

    # Handle Hydra instantiation pattern: Dict[str, Any] can represent any object
    if is_hydra_dict_type(config_type):
        return True

    # Handle List types
    if get_origin(config_type) is list and get_origin(target_type) is list:
        config_args = get_args(config_type)
        target_args = get_args(target_type)

        # List[Dict[str, Any]] in config can match List[AnyClass] in target
        if len(config_args) == 1 and len(target_args) == 1:
            if is_hydra_dict_type(config_args[0]):
                return True
            return is_compatible_type(config_args[0], target_args[0])

    # Handle other container types (dict, tuple, etc.)
    if get_origin(config_type) is get_origin(target_type):
        config_args = get_args(config_type)
        target_args = get_args(target_type)

        if len(config_args) != len(target_args):
            return False

        return all(is_compatible_type(c_arg, t_arg) for c_arg, t_arg in zip(config_args, target_args, strict=False))

    # Special case for int/float compatibility
    if config_type is int and target_type is float:
        return True

    return False


def resolve_callable_from_string(callable_str):
    """Resolve a string reference to a callable into the actual callable object."""
    try:
        module_path, attr_name = callable_str.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError, ValueError):
        return None


def is_callable_reference(value, expected_callable):
    """Check if a string value is a reference to the expected callable."""
    if not isinstance(value, str):
        return False

    # Resolve the string to a callable
    resolved_callable = resolve_callable_from_string(value)
    if resolved_callable is None:
        return False

    # Compare the resolved callable to the expected callable
    return resolved_callable == expected_callable


def compare_complex_default(config_default, param_default):
    """Safely compare complex default values (lists, dicts, etc.)."""
    # Special case: string reference to a callable in config
    if isinstance(config_default, str) and callable(param_default):
        # Try to resolve the string to a callable and compare
        return is_callable_reference(config_default, param_default)

    # For lists and tuples, compare lengths and values
    if isinstance(config_default, (list, tuple)) and isinstance(param_default, (list, tuple)):
        if len(config_default) != len(param_default):
            return False
        return all(compare_complex_default(c, p) for c, p in zip(config_default, param_default, strict=False))

    # For dictionaries, compare keys and values
    if isinstance(config_default, dict) and isinstance(param_default, dict):
        if set(config_default.keys()) != set(param_default.keys()):
            return False
        return all(compare_complex_default(config_default[k], param_default[k]) for k in config_default)

    # For numeric types, convert to float for comparison
    if isinstance(config_default, (int, float)) and isinstance(param_default, (int, float)):
        return float(config_default) == float(param_default)

    # Direct comparison for other types
    return config_default == param_default


def are_structurally_identical_types(type1, type2):
    """Check if two types look identical in string representation but have different internal objects."""
    if type1 == type2:
        return False  # They're already equal

    # Check if they have the same string representation
    return describe_type(type1) == describe_type(type2)


def test_config_classes_match_targets():
    """Test that config classes contain all parameters of their target classes with matching defaults."""
    # Register configs to make sure all are properly loaded
    register_configs()

    config_classes = get_config_classes()
    assert len(config_classes) > 0, "No config classes found"

    for config_class in config_classes:
        # Get the target class path
        target_path = None
        for field in fields(config_class):
            if field.name == "_target_":
                target_path = field.default
                break

        assert target_path is not None, f"Config class {config_class.__name__} has no _target_ field"

        # Skip if target begins with transformers or diffusers
        if target_path.startswith(("transformers.", "diffusers.")):
            continue

        # Import the target class
        try:
            target_class = import_target_class(target_path)
        except (ImportError, AttributeError) as e:
            # Skip tests for optional dependencies (e.g., Pi3 submodule)
            warnings.warn(f"Skipping {target_path}: dependency not available ({e})", UserWarning, stacklevel=1)
            continue

        # Get constructor parameters for both classes
        target_params = get_constructor_params(target_class)
        config_fields = {field.name: field for field in fields(config_class) if field.name != "_target_"}

        # Get the list of parameters that are injected externally
        injected_params = INJECTED_PARAMS.get(target_path, [])

        # Check that all target params are in config (except injected params)
        for param_name, param_default in target_params.items():
            # Skip parameters that are explicitly injected
            if param_name in injected_params:
                continue

            error_msg = f"Parameter '{param_name}' from {target_class.__name__} not found in {config_class.__name__}"
            assert param_name in config_fields, error_msg

            # Check type compatibility
            target_type_hints = get_type_hints(target_class.__init__)
            config_type_hints = get_type_hints(config_class)

            if param_name in target_type_hints and param_name in config_type_hints:
                target_type = target_type_hints[param_name]
                config_type = config_type_hints[param_name]

                if not is_compatible_type(config_type, target_type):
                    # Check if types are structurally identical but have different internals
                    looks_identical = are_structurally_identical_types(config_type, target_type)

                    error_msg = (
                        f"Type mismatch for '{param_name}' in {config_class.__name__}:\n"
                        f"  Config has: {describe_type(config_type)}\n"
                        f"  Target has: {describe_type(target_type)}\n"
                        f"  Config type id: {id(config_type)}, Origin: {get_origin(config_type)}, Args: {get_args(config_type)}\n"
                        f"  Target type id: {id(target_type)}, Origin: {get_origin(target_type)}, Args: {get_args(target_type)}\n"
                        f"  Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n"
                    )

                    if looks_identical:
                        error_msg += (
                            "\nNOTE: These types look identical but have different internal representations.\n"
                            "This is likely a false positive due to how Python's typing system works.\n"
                            "You can align the type definitions between config and implementation by\n"
                            "using the exact same syntax (e.g., if config uses Dict[str, float] but\n"
                            "implementation uses dict[str, float]). If this is not the case, you can\n"
                            "consider updating the test to handle this specific case if needed."
                        )
                    else:
                        error_msg += (
                            "This may be a legitimate Hydra configuration pattern. "
                            "If so, update the type compatibility check."
                        )

                    pytest.fail(error_msg)

            # Check default value compatibility with special cases for Hydra configs
            config_field = config_fields[param_name]
            config_default = config_field.default

            # If config default is not specified (MISSING), skip default value check
            if hasattr(config_default, "__name__") and config_default.__name__ == "MISSING":
                continue

            # For Hydra configs, empty lists and None can be equivalent
            field_type = config_type_hints.get(param_name)

            # Skip default value check for list/dict fields in Hydra configs
            if is_hydra_dict_type(field_type):
                continue

            if get_origin(field_type) is list:
                list_args = get_args(field_type)
                if len(list_args) == 1 and is_hydra_dict_type(list_args[0]):
                    continue

            # Special case for Any fields with default factory
            if field_type is Any and hasattr(config_field, "default_factory"):
                continue

            # Compare defaults using our helper function
            if param_default is not None and config_default is not None:
                # Special case: in Hydra configs, empty list may replace None
                if param_default is None and isinstance(config_default, list) and len(config_default) == 0:
                    continue

                # Special case: in Hydra configs, default factory for empty list
                if hasattr(config_default, "__name__") and config_default.__name__ == "default_factory":
                    continue

                # Skip MISSING values which typically indicate a default_factory is used
                if hasattr(config_default, "__class__") and config_default.__class__.__name__ == "_MISSING_TYPE":
                    continue

                if not compare_complex_default(config_default, param_default):
                    error_msg = (
                        f"Default value mismatch for '{param_name}' in {config_class.__name__}:\n"
                        f"  Config has: {config_default} (type: {type(config_default)})\n"
                        f"  Target has: {param_default} (type: {type(param_default)})"
                    )

                    # Add more info for string-to-callable failures
                    if isinstance(config_default, str) and callable(param_default):
                        resolved = resolve_callable_from_string(config_default)
                        if resolved is None:
                            error_msg += f"\n  Failed to resolve string '{config_default}' to a callable"
                        else:
                            error_msg += f"\n  String resolved to: {resolved} (different from target callable)"

                    error_msg += (
                        "\nNOTE: This could be a false positive if:\n"
                        "1. It's a string reference to a callable (for Hydra instantiation)\n"
                        "2. The default values are logically equivalent but not identical\n"
                        "3. Default is initialized through Hydra instantiation\n"
                        "If any of these apply, you can update the test to handle this case."
                        "\n\nRecommendation: Whenever possible, align the default values between config and implementation "
                        "to be identical. This simplifies configuration management and helps maintain compatibility."
                    )

                    pytest.fail(error_msg)


def test_no_extra_params_in_config():
    """Test that config classes don't have parameters that aren't in their target classes."""
    # Register configs to make sure all are properly loaded
    register_configs()

    config_classes = get_config_classes()

    for config_class in config_classes:
        # Get the target class path
        target_path = None
        for field in fields(config_class):
            if field.name == "_target_":
                target_path = field.default
                break

        # Skip if target begins with transformers or diffusers
        if target_path is None or target_path.startswith(("transformers.", "diffusers.")):
            continue

        # Import the target class
        try:
            target_class = import_target_class(target_path)
        except (ImportError, AttributeError):
            continue  # Skip if can't import

        # Get constructor parameters for target class
        target_params = get_constructor_params(target_class)

        # If the target class accepts **kwargs, all extra parameters are valid
        accepts_kwargs = False
        signature = inspect.signature(target_class.__init__)
        for _, param in signature.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:  # **kwargs
                accepts_kwargs = True
                break

        # Check that all config fields are in target params (except _target_)
        for field in fields(config_class):
            if field.name == "_target_":
                continue

            # Special cases where we allow fields in config not in target
            # These are typically for configuration or Hydra purposes
            special_cases = ["_recursive_", "_convert_"]
            if field.name in special_cases:
                continue

            # Skip params meant for Hydra instantiation
            field_type = get_type_hints(config_class).get(field.name)

            # If it's a Dict[str, Any] or List[Dict[str, Any]], it's likely for instantiation
            is_hydra_type = is_hydra_dict_type(field_type)

            # Check for List[Dict[str, Any]]
            if get_origin(field_type) is list:
                list_args = get_args(field_type)
                if len(list_args) == 1 and is_hydra_dict_type(list_args[0]):
                    is_hydra_type = True

            # If class accepts **kwargs or field is a Hydra instantiation type, skip the check
            if accepts_kwargs or is_hydra_type:
                continue

            # The config might contain parameters not in target_params because they're passed to other objects during instantiation
            # We check if this is a known pattern by looking for configs that are explicitly tied to instantiation patterns
            if field.name in ["model", "data"] and target_path == "dvlt.engine.trainer.Trainer":
                continue  # These are defined in the main Config class and passed during instantiation

            error_msg = (
                f"Parameter '{field.name}' exists in {config_class.__name__} "
                f"but not in {target_class.__name__}. This could lead to "
                f"configuration parameters being silently ignored."
            )
            assert field.name in target_params, error_msg
