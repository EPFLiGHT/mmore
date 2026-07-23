"""Global registry mapping tool names to callables.

Tools are registered programmatically (typically via the ``@register_tool``
decorator) and referenced from agent YAML configs by name.
"""

from typing import Callable

tool_registry: dict[str, Callable] = {}


class ToolNotRegisteredError(KeyError):
    """Raised when an agent config references a tool name that was never registered."""


def register_tool(name: str, fn: Callable | None = None) -> Callable:
    """Register ``fn`` under ``name``. Usable as a decorator or direct call.

    Examples:
        >>> @register_tool("greet")
        ... def greet(who: str) -> str: ...

        >>> register_tool("greet", greet_fn)
    """
    if fn is not None:
        tool_registry[name] = fn
        return fn

    def decorator(f: Callable) -> Callable:
        tool_registry[name] = f
        return f

    return decorator


def resolve_tool(name: str) -> Callable:
    """Resolve a single registered tool name into its callable."""
    try:
        return tool_registry[name]
    except KeyError:
        raise ToolNotRegisteredError(
            f"Tool '{name}' is not registered. Available tools: {sorted(tool_registry)}"
        ) from None


def resolve_tools(names: list[str]) -> list[Callable]:
    """Resolve a list of tool names into callables."""
    return [resolve_tool(name) for name in names]
