import functools
import inspect
import json
import logging
import threading
from collections.abc import Callable
from typing import Any

from mcp.server import MCPServer

from reaper_mcp import (
    analysis_tools,
    audio_tools,
    audio_view,
    compose_tools,
    fx_tools,
    loop_library,
    mastering_tools,
    midi_tools,
    mixing_tools,
    project_tools,
    region_render,
    render_tools,
    sampler_tools,
    template_tools,
    track_tools,
)

logger = logging.getLogger("reaper_mcp.server")

_server = MCPServer("reaper-mcp")

# reapy drives REAPER over a single shared socket: reapy.tools.network.Client
# keeps one module-global connection and its request() does an unguarded
# send-then-recv round trip. Two concurrent calls interleave their length
# prefixes and payloads on that socket and can read each other's replies.
#
# Under the old mcp.server.fastmcp this was unreachable, because v1 invoked
# sync tool functions inline on the event loop. MCP v2 dispatches them through
# anyio.to_thread.run_sync instead, so any client issuing parallel tool calls
# would run two of our tools on different worker threads at once. Every tool
# here is a sync def that ends up in reapy, so serialize them all behind one
# lock. This matches the effective v1 behaviour (one REAPER call at a time)
# without blocking the event loop.
_reaper_lock = threading.Lock()


def compact(result: dict) -> str:
    """A tool result as one line of JSON: the model reads every character, and
    the SDK's default (indent=2, plus a structured copy) roughly doubles it."""
    return json.dumps(result, separators=(",", ":"), ensure_ascii=False, default=str)


def _serialized(fn: Callable[..., dict]) -> Callable[..., str]:
    """Run the tool under the REAPER lock, reporting failure in the result.

    A tool's result is a dict with "success"; an exception becomes
    {"success": False, "error": str(e)} instead of an MCP error, which is the
    contract every tool here has always had. functools.wraps keeps the
    signature, which the SDK reads to build the tool's input schema.
    """

    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> str:
        with _reaper_lock:
            try:
                return compact(fn(*args, **kwargs))
            except Exception as e:
                logger.error("%s failed: %s", fn.__name__, e)
                return compact({"success": False, "error": str(e)})

    return run


mcp = _server

TOOL_MODULES = (
    project_tools,
    track_tools,
    midi_tools,
    fx_tools,
    loop_library,
    audio_tools,
    mixing_tools,
    render_tools,
    region_render,
    mastering_tools,
    analysis_tools,
    template_tools,
    audio_view,
    compose_tools,
    sampler_tools,
)

for _module in TOOL_MODULES:
    for _tool in _module.TOOLS:
        # cleandoc: the SDK sends the docstring verbatim, source indentation included.
        _server.tool(description=inspect.cleandoc(_tool.__doc__ or ""), structured_output=False)(
            _serialized(_tool))
