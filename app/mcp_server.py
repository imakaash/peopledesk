"""MCP server exposing the HR system (data/hr.db) as tools.

It runs as its own process and talks MCP over stdio. app/mcp_client.py starts it, so
you never launch it by hand. To poke at it directly:
    npx @modelcontextprotocol/inspector python -m app.mcp_server

SECURITY BOUNDARY. Every tool takes `auth_token`, the logged-in user's signed JWT.
The client injects it, and the model never sees or chooses it. Before touching data,
each tool calls _authorize(), which:
  1. verifies the token's signature and expiry  -> who is calling
  2. checks the role may use this tool          -> e.g. only managers approve leave
  3. checks the caller may touch the target's data -> own data, direct reports' leave, or HR
The client also hides tools the role can't use, but that is only for convenience.
These checks are the ones that count, because a prompt-injected or confused model
can ask for anything.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app import auth

DB = Path(__file__).resolve().parent.parent / "data" / "hr.db"
mcp = FastMCP("peopledesk-hr", log_level="WARNING")  # INFO logs every request to stderr

AuthToken = Annotated[str, Field(description="Set by the client, not by the model.")]
Employee = Annotated[str | None, Field(description="Employee id, e.g. 'dave'. Omit for the current user.")]


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _one(sql: str, *args) -> dict | None:
    with closing(_db()) as db:
        row = db.execute(sql, args).fetchone()
    return dict(row) if row else None


def _all(sql: str, *args) -> list[dict]:
    with closing(_db()) as db:
        return [dict(r) for r in db.execute(sql, args).fetchall()]


def _employee(emp_id: str) -> dict:
    emp = _one("SELECT * FROM employees WHERE id = ?", emp_id)
    if not emp:
        raise ValueError(f"No employee with id '{emp_id}'. Use search_employees to find ids.")
    return emp


def _authorize(token: str, tool: str, target: str | None = None, scope: str | None = None) -> tuple[str, dict]:
    """Return (target employee id, caller claims), or raise AuthError."""
    claims = auth.verify_token(token)
    if tool not in auth.allowed_tools(claims["role"]):
        raise auth.AuthError(f"Permission denied: the {claims['role']} role can't use {tool}.")
    target = target or claims["sub"]
    if scope and not auth.can_access(claims, target, scope):
        # Same message whether or not the employee exists, so ids can't be probed.
        raise auth.AuthError(f"Permission denied: you can't access {scope} data for '{target}'.")
    return target, claims


def _working_days(start: date, end: date) -> int:
    return sum(1 for i in range((end - start).days + 1) if (start + timedelta(days=i)).weekday() < 5)


def _balance(emp_id: str, year: int) -> dict:
    b = _one("SELECT * FROM leave_balances WHERE employee_id = ? AND year = ?", emp_id, year)
    if not b:
        raise ValueError(f"No {year} leave balance on record for '{emp_id}'.")
    pending = _one(
        "SELECT COALESCE(SUM(days), 0) AS d FROM leave_requests "
        "WHERE employee_id = ? AND status = 'pending' AND leave_type = 'annual' AND start_date LIKE ?",
        emp_id, f"{year}-%",
    )["d"]
    total = b["entitlement"] + b["carried_over"]
    return {
        "employee": emp_id, "year": year,
        "annual_entitlement": b["entitlement"], "carried_over": b["carried_over"],
        "annual_used": b["used"], "annual_pending": pending,
        "annual_available": total - b["used"] - pending,
        "sick_days_used": b["sick_used"], "sick_days_allowance": 10,
    }


@mcp.tool()
def get_leave_balance(auth_token: AuthToken, employee: Employee = None, year: int | None = None) -> dict:
    """Get annual and sick leave balance for an employee (default: the current user, current year).
    annual_available already subtracts pending requests. Managers can check their direct reports."""
    emp, _ = _authorize(auth_token, "get_leave_balance", employee, scope="leave")
    return _balance(emp, year or date.today().year)


@mcp.tool()
def apply_leave(
    auth_token: AuthToken,
    start_date: Annotated[str, Field(description="First day of leave, YYYY-MM-DD")],
    end_date: Annotated[str, Field(description="Last day of leave (inclusive), YYYY-MM-DD")],
    leave_type: Literal["annual", "sick", "parental", "unpaid", "compassionate"] = "annual",
    note: str = "",
) -> dict:
    """Submit a leave request for the current user. It is created as 'pending' for their manager to approve.
    Enforces policy: no past dates for annual leave, enough balance, and 2 weeks' notice for more than 3 working days."""
    me, _ = _authorize(auth_token, "apply_leave")  # always for yourself; no `employee` parameter on purpose
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    today = date.today()
    if end < start:
        raise ValueError("end_date is before start_date.")
    days = _working_days(start, end)
    if days == 0:
        raise ValueError("That range has no working days.")
    if leave_type == "annual":
        if start < today:
            raise ValueError("Annual leave can't start in the past.")
        if days > 3 and (start - today).days < 14:
            raise ValueError(f"{days} working days needs 2 weeks' notice; the earliest start is {today + timedelta(days=14)}.")
        available = _balance(me, start.year)["annual_available"]
        if days > available:
            raise ValueError(f"Not enough annual leave: requested {days} days, {available} available.")
    with closing(_db()) as db, db:
        cur = db.execute(
            "INSERT INTO leave_requests (employee_id, leave_type, start_date, end_date, days, note, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (me, leave_type, start_date, end_date, days, note, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
    return {"request_id": cur.lastrowid, "status": "pending", "working_days": days, "approver": _employee(me)["manager_id"]}


@mcp.tool()
def list_leave_requests(
    auth_token: AuthToken, employee: Employee = None,
    status: Literal["pending", "approved", "declined"] | None = None,
    team: Annotated[bool, Field(description="True: requests from the caller's direct reports instead")] = False,
) -> list[dict]:
    """List leave requests: the current user's own, a direct report's, or (team=True) all of the caller's direct reports."""
    if team:
        _, claims = _authorize(auth_token, "list_leave_requests")
        # Scoped by the query itself: only rows whose manager is the caller.
        sql, args = "SELECT r.* FROM leave_requests r JOIN employees e ON e.id = r.employee_id WHERE e.manager_id = ?", [claims["sub"]]
    else:
        emp, _ = _authorize(auth_token, "list_leave_requests", employee, scope="leave")
        sql, args = "SELECT * FROM leave_requests r WHERE r.employee_id = ?", [emp]
    if status:
        sql += " AND r.status = ?"
        args.append(status)
    return _all(sql + " ORDER BY r.start_date", *args)


@mcp.tool()
def approve_leave(
    auth_token: AuthToken,
    request_id: int,
    decision: Literal["approve", "decline"],
    reason: Annotated[str, Field(description="Required when declining")] = "",
) -> dict:
    """Approve or decline a pending leave request from one of your direct reports. Approved annual leave is deducted from the balance."""
    _, claims = _authorize(auth_token, "approve_leave")
    req = _one("SELECT * FROM leave_requests WHERE id = ?", request_id)
    if not req:
        raise ValueError(f"No leave request #{request_id}.")
    if req["employee_id"] == claims["sub"]:
        raise auth.AuthError("Permission denied: you can't approve your own leave.")
    _authorize(auth_token, "approve_leave", req["employee_id"], scope="leave")  # must be their manager (or HR)
    if req["status"] != "pending":
        raise ValueError(f"Request #{request_id} is already {req['status']}.")
    if decision == "decline" and not reason.strip():
        raise ValueError("Policy requires a reason when declining leave.")
    status = "approved" if decision == "approve" else "declined"
    with closing(_db()) as db, db:
        db.execute(
            "UPDATE leave_requests SET status = ?, decided_by = ?, decision_reason = ? WHERE id = ?",
            (status, claims["sub"], reason, request_id),
        )
        if status == "approved" and req["leave_type"] == "annual":
            db.execute(
                "UPDATE leave_balances SET used = used + ? WHERE employee_id = ? AND year = ?",
                (req["days"], req["employee_id"], int(req["start_date"][:4])),
            )
    return {"request_id": request_id, "status": status, "employee": req["employee_id"]}


@mcp.tool()
def get_payslip(
    auth_token: AuthToken,
    month: Annotated[str, Field(description="YYYY-MM")],
    employee: Employee = None,
) -> dict:
    """Get the payslip for one month: gross pay, deductions and net pay, in EUR. Employees see only their own."""
    emp, _ = _authorize(auth_token, "get_payslip", employee, scope="pay")  # managers don't get "pay" scope
    slip = _one("SELECT * FROM payslips WHERE employee_id = ? AND month = ?", emp, month)
    if not slip:
        months = [r["month"] for r in _all("SELECT month FROM payslips WHERE employee_id = ? ORDER BY month", emp)]
        raise ValueError(f"No payslip for {emp} in {month}. Available months: {months or 'none yet'}.")
    return slip


@mcp.tool()
def list_onboarding_tasks(auth_token: AuthToken, employee: Employee = None) -> list[dict]:
    """List onboarding tasks with due dates and whether they are done. Overdue = not done and due_date before today."""
    emp, _ = _authorize(auth_token, "list_onboarding_tasks", employee, scope="onboarding")
    tasks = _all("SELECT task, due_date, done FROM onboarding_tasks WHERE employee_id = ? ORDER BY due_date", emp)
    today = date.today().isoformat()
    return [{**t, "done": bool(t["done"]), "overdue": not t["done"] and t["due_date"] < today} for t in tasks]


@mcp.tool()
def book_hr_meeting(
    auth_token: AuthToken,
    day: Annotated[str, Field(description="YYYY-MM-DD, a working day")],
    hour: Annotated[int, Field(ge=9, le=16, description="Start hour, 9 to 16. Meetings are 1 hour.")],
    topic: str,
) -> dict:
    """Book a 1-hour 1:1 with the HR team for the current user."""
    me, _ = _authorize(auth_token, "book_hr_meeting")
    d = date.fromisoformat(day)
    if d <= date.today() or d.weekday() >= 5:
        raise ValueError("Pick a future working day (Monday to Friday).")
    slot = f"{day}T{hour:02d}:00"
    if _one("SELECT 1 AS x FROM hr_meetings WHERE slot = ?", slot):
        raise ValueError(f"{slot} is already booked. Try another hour.")
    with closing(_db()) as db, db:
        db.execute(
            "INSERT INTO hr_meetings (employee_id, slot, topic, created_at) VALUES (?, ?, ?, ?)",
            (me, slot, topic, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
    return {"booked": slot, "with": "HR team (carol@nimbuslabs.example)", "topic": topic}


@mcp.tool()
def search_employees(auth_token: AuthToken, query: Annotated[str, Field(description="Name, team or job title")]) -> list[dict]:
    """Search the company directory. Returns ids, names, titles, teams, emails and managers (no private data)."""
    _authorize(auth_token, "search_employees")  # directory data is visible to every logged-in user
    q = f"%{query.lower()}%"
    return _all(
        "SELECT id, name, title, team, email, manager_id FROM employees "
        "WHERE lower(name) LIKE ? OR lower(team) LIKE ? OR lower(title) LIKE ? OR id LIKE ?",
        q, q, q, q,
    )


if __name__ == "__main__":
    mcp.run()  # stdio transport
