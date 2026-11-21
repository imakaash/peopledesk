"""Shared test setup: a test JWT secret and a fresh, seeded HR database per test."""
import importlib.util
import os

import pytest

# Set before anything reads it. load_dotenv() never overrides an existing env var,
# so tests don't depend on (or touch) your .env.
os.environ["JWT_SECRET"] = "test-secret-for-pytest-only-0123456789abcdef"
os.environ["DEMO_PASSWORD"] = "nimbus123"

from app import auth, llm, mcp_server, tracing  # noqa: E402

# Tests never send spans to Phoenix, even when it's running.
tracing._state.update(initialized=True, enabled=False)

_spec = importlib.util.spec_from_file_location("seed", llm.ROOT / "data" / "seed.py")
seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed)


@pytest.fixture(autouse=True)
def hr_db(tmp_path, monkeypatch):
    db = tmp_path / "hr.db"
    monkeypatch.setattr(seed, "DB", db)
    monkeypatch.setattr(mcp_server, "DB", db)
    monkeypatch.setattr(auth, "HR_DB", db)
    seed.main()
    return db


def token_for(user_id: str) -> str:
    emp = auth._employee(user_id)
    return auth.issue_token(emp["id"], emp["name"], emp["role"])
