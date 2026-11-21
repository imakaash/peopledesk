"""RBAC tests: call the MCP tool functions directly, like a hostile client could.

No LLM is involved. These check the server-side rules hold whatever the model asks for.
"""
from datetime import date, timedelta

import jwt
import pytest

from app import auth, rag
from app import mcp_server as hr
from conftest import token_for

Denied = auth.AuthError


def next_weekday(days_ahead: int) -> str:
    d = date.today() + timedelta(days=days_ahead)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.isoformat()


# --- authN: tokens ----------------------------------------------------------------

def test_login_success_and_failure():
    claims = auth.verify_token(auth.login("Alice", "nimbus123"))
    assert claims["sub"] == "alice" and claims["role"] == "employee"
    with pytest.raises(Denied):
        auth.login("alice", "wrong")
    with pytest.raises(Denied):
        auth.login("mallory", "nimbus123")


def test_forged_token_rejected():
    # Signed with a different secret, claiming to be HR: signature check must fail.
    forged = jwt.encode({"sub": "alice", "role": "hr_admin", "iss": "peopledesk", "exp": 9999999999}, "x" * 40, algorithm="HS256")
    with pytest.raises(Denied):
        hr.get_payslip(forged, "2026-10", employee="dave")


def test_unsigned_alg_none_token_rejected():
    unsigned = jwt.encode({"sub": "carol", "role": "hr_admin", "iss": "peopledesk", "exp": 9999999999}, None, algorithm="none")
    with pytest.raises(Denied):
        hr.get_payslip(unsigned, "2026-10", employee="dave")


def test_expired_token_rejected():
    expired = auth.issue_token("dave", "Dave", "employee", ttl=timedelta(seconds=-1))
    with pytest.raises(Denied, match="expired"):
        hr.get_leave_balance(expired)


# --- authZ: whose data ------------------------------------------------------------

def test_employee_sees_own_payslip_only():
    assert hr.get_payslip(token_for("dave"), "2026-10")["net"] > 0
    with pytest.raises(Denied):
        hr.get_payslip(token_for("alice"), "2026-10", employee="dave")


def test_manager_sees_reports_leave_but_not_pay():
    bob = token_for("bob")
    assert hr.get_leave_balance(bob, employee="dave")["employee"] == "dave"
    with pytest.raises(Denied):
        hr.get_payslip(bob, "2026-10", employee="dave")  # managers have no "pay" scope


def test_manager_cannot_see_other_teams():
    with pytest.raises(Denied):
        hr.get_leave_balance(token_for("bob"), employee="erin")  # erin reports to frank


def test_hr_admin_sees_everything():
    assert hr.get_payslip(token_for("carol"), "2026-10", employee="dave")["employee_id"] == "dave"


# --- authZ: what actions ----------------------------------------------------------

def test_employee_cannot_approve_leave():
    with pytest.raises(Denied, match="employee role"):
        hr.approve_leave(token_for("alice"), request_id=1, decision="approve")


def test_manager_approves_only_own_reports():
    bob = token_for("bob")
    assert hr.approve_leave(bob, request_id=1, decision="approve")["status"] == "approved"  # dave's
    with pytest.raises(Denied):
        hr.approve_leave(bob, request_id=2, decision="approve")  # erin's, frank's team


def test_nobody_approves_their_own_leave():
    carol = token_for("carol")
    day = next_weekday(30)
    req = hr.apply_leave(carol, start_date=day, end_date=day)
    with pytest.raises(Denied, match="own leave"):
        hr.approve_leave(carol, request_id=req["request_id"], decision="approve")


def test_apply_leave_is_always_for_the_caller():
    day = next_weekday(30)
    req = hr.apply_leave(token_for("alice"), start_date=day, end_date=day)
    assert hr.list_leave_requests(token_for("alice"), status="pending")[0]["id"] == req["request_id"]


# --- layer 1: what the model is shown ---------------------------------------------

def test_tool_listing_per_role():
    assert "approve_leave" not in auth.allowed_tools("employee")
    assert "approve_leave" in auth.allowed_tools("manager")


# --- documents --------------------------------------------------------------------

def test_hr_docs_hidden_from_non_hr():
    if rag._collection().count() == 0:
        pytest.skip("RAG index is empty; run `python -m app.rag ingest`")
    for role in ("employee", "manager"):
        hits = rag.retrieve("salary bands bonus payroll", access=auth.doc_access(role))
        assert all(h["access"] == "all" for h in hits)
    hr_hits = rag.retrieve("salary bands", access=auth.doc_access("hr_admin"))
    assert any(h["source"] == "payroll.md" for h in hr_hits)
