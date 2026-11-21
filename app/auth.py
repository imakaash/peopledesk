"""Authentication (who are you?) and authorization (what may you do?).

authN  login() checks credentials and issues a signed JWT: {sub, name, role, exp}.
       Anyone holding the secret can mint tokens, so JWT_SECRET must stay private.
authZ  The rules below decide which tools a role may call, which documents it may
       read, and whose data it may touch. They are enforced in two places:
         - mcp_client.py hides tools the role can't use (less confusion, fewer tokens)
         - mcp_server.py re-checks every call (the real security boundary)
       The model is never trusted: it doesn't choose identities, and when it's
       wrong, the server just refuses.

The local login is a demo stand-in. In production, an identity provider (OIDC:
Google, Okta, Entra) authenticates the user, and we'd map their email to a role.
Streamlit's st.login() can do that part.
"""
from __future__ import annotations

import hmac
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

HR_DB = ROOT / "data" / "hr.db"
ISSUER = "peopledesk"
TOKEN_TTL = timedelta(hours=8)

EMPLOYEE_TOOLS = {
    "get_leave_balance", "apply_leave", "list_leave_requests", "get_payslip",
    "list_onboarding_tasks", "book_hr_meeting", "search_employees",
}
ROLES = {
    #            tools the role may call               RAG doc tags it may read
    "employee": {"tools": EMPLOYEE_TOOLS,                  "docs": ["all"]},
    "manager":  {"tools": EMPLOYEE_TOOLS | {"approve_leave"}, "docs": ["all"]},
    "hr_admin": {"tools": EMPLOYEE_TOOLS | {"approve_leave"}, "docs": ["all", "hr"]},
}
# Which kinds of a direct report's data a manager may see. Pay is deliberately missing.
MANAGER_SCOPES = {"leave", "onboarding"}


class AuthError(PermissionError):
    pass


def jwt_secret() -> str:
    s = os.environ.get("JWT_SECRET", "")
    if len(s) < 32 or s == "change-me":
        raise RuntimeError(
            "JWT_SECRET in .env must be a random string of at least 32 characters. Generate one with:\n"
            "  python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
    return s


def _employee(user_id: str) -> dict | None:
    with closing(sqlite3.connect(HR_DB)) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT id, name, role, manager_id FROM employees WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


# --- authN --------------------------------------------------------------------

def login(username: str, password: str) -> str:
    """Demo login: every seeded employee shares DEMO_PASSWORD. Returns a signed JWT."""
    emp = _employee(username.strip().lower())
    expected = os.environ.get("DEMO_PASSWORD", "nimbus123")
    # compare_digest takes the same time however many characters match (no timing leak)
    if not emp or not hmac.compare_digest(password.encode(), expected.encode()):
        raise AuthError("Wrong username or password.")  # same message either way: don't reveal which
    return issue_token(emp["id"], emp["name"], emp["role"])


def issue_token(user_id: str, name: str, role: str, ttl: timedelta = TOKEN_TTL) -> str:
    now = datetime.now(timezone.utc)
    claims = {"sub": user_id, "name": name, "role": role, "iss": ISSUER, "iat": now, "exp": now + ttl}
    return jwt.encode(claims, jwt_secret(), algorithm="HS256")


def verify_token(token: str) -> dict:
    """Check the signature, expiry and issuer. A changed or expired token raises AuthError."""
    try:
        # Pin the algorithm list: never let the token header choose it (alg=none attacks).
        claims = jwt.decode(token, jwt_secret(), algorithms=["HS256"], issuer=ISSUER, options={"require": ["exp", "sub", "role"]})
    except jwt.ExpiredSignatureError:
        raise AuthError("Session expired. Please log in again.")
    except jwt.InvalidTokenError as e:
        raise AuthError(f"Invalid session token ({e}).")
    if claims["role"] not in ROLES:
        raise AuthError("Unknown role in token.")
    return claims


# --- authZ --------------------------------------------------------------------

def allowed_tools(role: str) -> set[str]:
    return ROLES[role]["tools"]


def doc_access(role: str) -> list[str]:
    return ROLES[role]["docs"]


def can_access(claims: dict, target_id: str, scope: str) -> bool:
    """May this caller see or act on `scope` data ("leave", "onboarding", "pay") of `target_id`?"""
    if target_id == claims["sub"] or claims["role"] == "hr_admin":
        return True
    if claims["role"] == "manager" and scope in MANAGER_SCOPES:
        target = _employee(target_id)
        return bool(target and target["manager_id"] == claims["sub"])
    return False
