"""MCP client: connects to app/mcp_server.py over stdio and exposes its tools to Claude.

The MCP SDK is async, but Streamlit and our agent are sync. So one background thread
runs an asyncio loop that owns the MCP session, and the sync methods below send work
to it. The server subprocess starts once and is reused for every request.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
import threading
from dataclasses import dataclass
from functools import cache

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.llm import ROOT

HIDDEN_ARGS = {"caller"}  # filled in by the harness, never shown to or chosen by the model


@dataclass
class ToolResult:
    text: str
    is_error: bool


class MCPClient:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-loop").start()
        ready: concurrent.futures.Future = concurrent.futures.Future()
        asyncio.run_coroutine_threadsafe(self._serve(ready), self._loop)
        self._tools = ready.result(timeout=30)

    async def _serve(self, ready: concurrent.futures.Future) -> None:
        # The session lives inside this one long-running task, so its async context
        # managers are entered and exited in the same task, as anyio requires.
        params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], cwd=str(ROOT))
        try:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                ready.set_result((await session.list_tools()).tools)
                await asyncio.Event().wait()  # keep the session open for the life of the process
        except Exception as e:
            if not ready.done():
                ready.set_exception(e)
            raise

    def claude_tools(self) -> list[dict]:
        """MCP tool definitions -> Claude tool definitions.

        Sorted by name so the tools block is byte-identical on every request; any
        change there would break prompt caching. Hidden args are removed from the schema.
        """
        tools = []
        for t in sorted(self._tools, key=lambda t: t.name):
            schema = json.loads(json.dumps(t.inputSchema))  # deep copy
            for arg in HIDDEN_ARGS:
                schema.get("properties", {}).pop(arg, None)
                if arg in schema.get("required", []):
                    schema["required"].remove(arg)
            tools.append({"name": t.name, "description": t.description or "", "input_schema": schema})
        return tools

    def call(self, name: str, args: dict, caller: str) -> ToolResult:
        # The harness sets the identity. Anything the model put in `caller` is overwritten.
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, {**args, "caller": caller}), self._loop
        )
        try:
            res = future.result(timeout=30)
        except Exception as e:  # transport failure: report it to the model, don't crash the turn
            return ToolResult(f"Tool call failed: {e}", True)
        text = "\n".join(c.text for c in res.content if c.type == "text")
        return ToolResult(text or "(no output)", bool(res.isError))


@cache
def client() -> MCPClient:
    return MCPClient()


if __name__ == "__main__":
    for t in client().claude_tools():
        print(f"- {t['name']}: {list(t['input_schema'].get('properties', {}))}")
