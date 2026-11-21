"""Answer-cache safety rules. Uses a throwaway Chroma collection; the LLM verifier is stubbed."""
import pytest

from app import cache

Q = "How many sick days do employees get per year?"


@pytest.fixture(autouse=True)
def test_collection(monkeypatch):
    monkeypatch.setattr(cache, "COLLECTION", "answer_cache_pytest")
    monkeypatch.setattr(cache, "same_question", lambda a, b: True)
    cache.clear()
    yield
    cache.clear()


def test_hit_for_same_role():
    cache.store(Q, "employee", "10 paid sick days.")
    hit, reason = cache.lookup(Q, "employee")
    assert hit and hit["answer"] == "10 paid sick days." and reason == "hit"


def test_scoped_by_role():
    # An hr_admin answer may quote HR-only docs: it must never be served to an employee.
    cache.store(Q, "hr_admin", "HR-internal answer")
    hit, reason = cache.lookup(Q, "employee")
    assert hit is None and "role 'employee'" in reason


def test_doc_or_prompt_change_invalidates(monkeypatch):
    cache.store(Q, "employee", "old answer")
    monkeypatch.setattr(cache, "fingerprint", lambda: "docs-were-edited")
    assert cache.lookup(Q, "employee")[0] is None


def test_verifier_can_veto_a_close_match(monkeypatch):
    cache.store("How much parental leave does the primary caregiver get?", "employee", "16 weeks")
    monkeypatch.setattr(cache, "same_question", lambda a, b: False)
    hit, reason = cache.lookup("How much parental leave does the secondary caregiver get?", "employee")
    assert hit is None and reason.startswith("verifier")


def test_second_candidate_can_hit(monkeypatch):
    # The closest match is rejected, but the next one is the same question.
    cache.store("how many sick days do i get?", "employee", "A")
    cache.store("How many sick days do employees get?", "employee", "B")
    monkeypatch.setattr(cache, "same_question", lambda a, b: "employees" in b)
    hit, _ = cache.lookup("How many sick days do we get?", "employee")
    assert hit and hit["answer"] == "B"
