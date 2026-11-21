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
from mcp.client.stdio import get_default_environment, stdio_client

from app import auth
from app.llm import ROOT

HIDDEN_ARGS = {"auth_token"}  # filled in by the harness, never shown to or chosen by the model


@dataclass
class ToolResult:
    text: str
    is_error: bool


class MCPClient:
    def __init__(self, hr_db: str | None = None) -> None:
        """`hr_db` points the server at another database file (evals use a fresh copy per case)."""
        self._hr_db = hr_db
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-loop").start()
        ready: concurrent.futures.Future = concurrent.futures.Future()
        self._task = asyncio.run_coroutine_threadsafe(self._serve(ready), self._loop)
        self._tools = ready.result(timeout=30)

    async def _serve(self, ready: concurrent.futures.Future) -> None:
        # The session lives inside this one long-running task, so its async context
        # managers are entered and exited in the same task, as anyio requires.
        # The stdio client passes only a small safe set of env vars to the subprocess.
        # The server needs the same JWT secret to verify tokens, so pass it explicitly.
        env = {**get_default_environment(), "JWT_SECRET": auth.jwt_secret()}
        if self._hr_db:
            env["PEOPLEDESK_HR_DB"] = self._hr_db
        params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], cwd=str(ROOT), env=env)
        self._stop = asyncio.Event()
        try:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                ready.set_result((await session.list_tools()).tools)
                await self._stop.wait()  # stays open until close() (the app never calls it)
        except Exception as e:
            if not ready.done():
                ready.set_exception(e)
            raise

    def close(self) -> None:
        """Shut down the server subprocess and the loop thread (evals start one per case)."""
        self._loop.call_soon_threadsafe(self._stop.set)
        try:
            self._task.result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    def claude_tools(self, role: str) -> list[dict]:
        """MCP tool definitions -> Claude tool definitions, for one role.

        Layer 1 of RBAC: the model only sees tools the role may use. An employee's
        model doesn't know approve_leave exists. (Layer 2, the real check, is in the server.)
        Sorted by name so the tools block is byte-identical per role on every request;
        any change there would break prompt caching. Hidden args are removed from the schema.
        """
        allowed = auth.allowed_tools(role)
        tools = []
        for t in sorted(self._tools, key=lambda t: t.name):
            if t.name not in allowed:
                continue
            schema = json.loads(json.dumps(t.inputSchema))  # deep copy
            for arg in HIDDEN_ARGS:
                schema.get("properties", {}).pop(arg, None)
                if arg in schema.get("required", []):
                    schema["required"].remove(arg)
            tools.append({"name": t.name, "description": t.description or "", "input_schema": schema})
        return tools

    def call(self, name: str, args: dict, token: str) -> ToolResult:
        # The harness attaches the user's signed token. If the model tried to put an
        # auth_token in args, it's overwritten here.
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, {**args, "auth_token": token}), self._loop
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
    for role in auth.ROLES:
        print(role, "->", [t["name"] for t in client().claude_tools(role)])
