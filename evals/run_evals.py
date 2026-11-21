"""Run the golden set through the real agent, grade every answer, apply the release gate.

    python evals/run_evals.py                          # the live prompt (config.yaml prompt_version)
    python evals/run_evals.py --prompt v2              # a candidate prompt, without changing anything
    python evals/run_evals.py --prompt v2 --release    # ...and make it live if the gate passes
    python evals/run_evals.py --only policy-gifts rbac-hr-salary-bands   # a few cases (pilot)

Exit code: 0 = gate passed, 1 = gate failed, 2 = harness error (the eval itself broke).

This is the LLM Ops loop from the diagram: trace -> eval -> diagnose -> gate -> release.
Every result row carries its Phoenix trace_id, so a failing case leads straight to
the span that went wrong (project "peopledesk-evals").
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from app import llm  # noqa: E402  (loads config.yaml; nothing has touched memory or Chroma yet)

CFG = llm.CONFIG["evals"]


def load_seed():
    spec = importlib.util.spec_from_file_location("seed", ROOT / "data" / "seed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def isolate(run_dir: Path, prompt: str) -> None:
    """Point every store at fresh files inside run_dir, so a run never sees (or changes)
    your dev data and two runs never see each other. Must happen before first use."""
    c = llm.CONFIG
    c["prompt_version"] = prompt
    c["memory"]["db_path"] = str(run_dir / "memory.db")    # empty episodic memory and profiles
    c["memory"]["consolidate_every_n"] = 10**9              # no background summarizer calls
    # Cases must be independent. Cases run in parallel and share this run's memory DB, so
    # without these, one case's conversation ("I approved Dave's leave") shows up as a
    # past episode in another case (where Dave's leave is still pending), and results
    # change with the order the cases ran in. This was found when v1 swung from 84% to 70%.
    c["memory"]["recent_episodes"] = 0                      # no past episodes injected
    c["answer_cache"]["enabled"] = False                    # no answer reuse across cases (tests/test_cache.py covers it)
    c["rag"]["chroma_path"] = str(run_dir / "chroma")       # fresh doc index, no facts, empty answer cache
    c["tracing"]["project"] = "peopledesk-evals"            # keep eval traces apart from app traces
    from app import rag
    rag.ingest()


# --- deterministic checks ---------------------------------------------------------

def run_checks(case: dict, answer: str, result: dict) -> dict:
    exp, tags = case.get("expect", {}), set(case["tags"])
    called = [c["name"] for c in result.get("tool_calls", [])]
    blocked = result["reply"].stop_reason == "guardrail_blocked"
    failures, leak = [], False
    security_case = bool(tags & {"rbac", "leak-check", "injection"})

    for p in exp.get("facts", []):
        if not re.search(p, answer, re.I):
            failures.append(f"missing fact /{p}/")
    for p in exp.get("forbid", []):
        if m := re.search(p, answer, re.I):
            failures.append(f"forbidden /{p}/ matched '{m.group(0)}'")
            leak = leak or "leak-check" in tags
    for t in exp.get("tools_called", []):
        if t not in called:
            failures.append(f"tool {t} not called")
    for t in exp.get("tools_not_called", []):
        if t in called:
            failures.append(f"tool {t} called but must not be")
            leak = leak or security_case  # a forbidden *action* in a security case counts too
    if exp.get("blocked") is True and not blocked:
        failures.append("expected the input guardrail to block it")
    false_block = blocked and exp.get("blocked") is not True
    if false_block:
        failures.append("legitimate question was blocked")
    route_ok = None if "route" not in exp else result["meta"]["route"] == exp["route"]
    return {"failures": failures, "leak": leak, "false_block": false_block, "blocked": blocked,
            "route_ok": route_ok, "tools": called}


# --- one case -----------------------------------------------------------------------

def run_case(case: dict, run_dir: Path, seed, judge_mod) -> dict:
    from app import auth
    from app.agent import Agent
    from app.mcp_client import MCPClient

    db = run_dir / "db" / f"{case['id']}.db"
    seed.main(db, verbose=False)  # fresh HR data: cases that book or approve leave can't affect each other
    emp = auth._employee(case["user"])
    token = auth.issue_token(emp["id"], emp["name"], emp["role"])

    mcp = MCPClient(hr_db=str(db))
    try:
        t0, result = time.perf_counter(), {}
        answer_stream = Agent(token, f"eval-{case['id']}", mcp=mcp).stream(case["question"], case.get("history", []), result)
        "".join(answer_stream)
        latency = time.perf_counter() - t0
    finally:
        mcp.close()

    answer = result["reply"].text
    checks = run_checks(case, answer, result)
    if checks["blocked"] and case["expect"].get("blocked") is True:
        j = {"grades": {"correct": 1.0, "safe": 1.0, "tone": 1.0}, "reasons": {"correct": "blocked as expected"},
             "model": None, "usage": {}, "cost_usd": 0.0}  # no need to pay a judge for the safe reply
    else:
        j = judge_mod.judge(case, emp["role"], answer)

    # Your pilot call: an answer the app itself flagged as ungrounded (the ⚠️ banner) fails,
    # even when the key fact happens to be right. It usually means retrieval missed the doc.
    app_grounded = getattr(result.get("guard_out"), "grounded", None)
    if app_grounded is False:
        claims = getattr(result.get("guard_out"), "unsupported_claims", [])
        checks["failures"].append(f"app flagged its own answer as ungrounded: {claims}")
    ok = not checks["failures"]
    m = result["meta"]
    return {
        "prompt_id": case["id"], "tags": case["tags"], "user": case["user"], "role": emp["role"],
        "prompt": case["question"], "answer": answer, "status": "ok",
        "grade": {
            "pass": float(ok and j["grades"]["correct"] == 1 and j["grades"]["safe"] == 1),
            "checks": float(ok), **j["grades"],
            "leak": float(checks["leak"]), "false_block": float(checks["false_block"]),
        },
        "explanation": {"checks": "; ".join(checks["failures"]) or "all passed", **j["reasons"]},
        "route": m["route"], "route_ok": checks["route_ok"], "model": m["model"],
        "tool_calls": checks["tools"], "app_grounded": app_grounded,
        "app_unsupported_claims": getattr(result.get("guard_out"), "unsupported_claims", []),
        "latency_s": round(latency, 2), "app_cost_usd": m.get("cost_usd_total", 0.0),
        "judge_model": j["model"], "judge_usage": j["usage"], "judge_cost_usd": j["cost_usd"],
        "trace_id": m.get("trace_id"),
    }


# --- summary + gate -----------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a pass rate. With few cases it's wide, and that's the honest answer."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def summarize(rows: list[dict], errors: list[dict]) -> dict:
    n = len(rows)
    passed = sum(r["grade"]["pass"] for r in rows)
    lo, hi = wilson(int(passed), n)
    groups = defaultdict(list)
    for r in rows:
        groups[r["tags"][0]].append(r["grade"]["pass"])
    routed = [r["route_ok"] for r in rows if r["route_ok"] is not None]
    lat = [r["latency_s"] for r in rows]
    return {
        "cases": n, "errors": len(errors), "pass_rate": passed / n if n else 0.0, "pass_ci95": [lo, hi],
        "leaks": int(sum(r["grade"]["leak"] for r in rows)),
        "false_blocks": int(sum(r["grade"]["false_block"] for r in rows)),
        "by_group": {g: f"{int(sum(v))}/{len(v)}" for g, v in sorted(groups.items())},
        "judge": {k: sum(r["grade"][k] for r in rows) / n for k in ("correct", "safe", "tone")} if n else {},
        "routing_accuracy": f"{sum(routed)}/{len(routed)}" if routed else "n/a",
        "latency_s": {"median": statistics.median(lat), "max": max(lat)} if lat else {},
        "cost_usd": {"app": sum(r["app_cost_usd"] for r in rows), "judge": sum(r["judge_cost_usd"] for r in rows)},
    }


def reference_run(live: str, current: Path, case_ids: set[str]) -> tuple[Path | None, dict]:
    """The live prompt's most recent complete run on the same cases: what a candidate is compared with.

    Not "the last run that passed the gate": that picked a stale run from before a harness
    fix and an older case set, and reported regressions that weren't real.
    """
    for d in sorted((ROOT / "evals" / "reports").glob(f"*-{live}"), reverse=True):
        results = d / "results.jsonl"
        if d == current or not results.exists():
            continue
        rows = {r["prompt_id"]: r for r in map(json.loads, results.read_text().splitlines())}
        if case_ids <= rows.keys():
            return d, rows
    return None, {}


def regressions(rows: list[dict], reference: dict) -> list[str]:
    """Cases in protected groups that passed on the live prompt and fail on the candidate.
    A pass-rate threshold can't see these: one lost safety case moves the average 3 points."""
    groups = set(CFG["gate"]["no_regression_groups"])
    return sorted(
        r["prompt_id"] for r in rows
        if r["tags"][0] in groups and not r["grade"]["pass"]
        and reference.get(r["prompt_id"], {}).get("grade", {}).get("pass") == 1.0
    )


def gate(s: dict, regressed: list[str] | None = None, missing_reference: str | None = None) -> list[str]:
    g, why = CFG["gate"], []
    if missing_reference:
        why.append(missing_reference)
    if regressed:
        why.append(f"regression vs live prompt in {', '.join(regressed)}")
    if s["errors"]:
        why.append(f"{s['errors']} case(s) errored: fix the harness before trusting this run")
    if s["pass_rate"] < g["min_pass_rate"]:
        why.append(f"pass rate {s['pass_rate']:.0%} < {g['min_pass_rate']:.0%}")
    if s["leaks"] > g["max_leaks"]:
        why.append(f"{s['leaks']} leak(s) > {g['max_leaks']}")
    if s["false_blocks"] > g["max_false_blocks"]:
        why.append(f"{s['false_blocks']} false block(s) > {g['max_false_blocks']}")
    return why


def write_report(run_dir: Path, prompt: str, s: dict, failures: list[str], rows: list[dict], errors: list[dict]) -> Path:
    lines = [
        f"# Eval report: prompt `{prompt}`, {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"**Gate: {'PASSED ✅' if not failures else 'FAILED ❌'}**" + ("" if not failures else " - " + "; ".join(failures)),
        "",
        f"- Pass rate: **{s['pass_rate']:.0%}** ({int(s['pass_rate'] * s['cases'])}/{s['cases']}), 95% CI {s['pass_ci95'][0]:.0%}-{s['pass_ci95'][1]:.0%}",
        f"- Leaks: **{s['leaks']}**, false blocks: **{s['false_blocks']}**, errors: {s['errors']}",
        f"- Regressions vs live prompt: **{len(s.get('regressions', []))}** {s.get('regressions') or ''}"
        + (f" (reference: `{s['reference_run']}`)" if s.get("reference_run") else " (not a candidate run)"),
        f"- By group: {', '.join(f'{g} {v}' for g, v in s['by_group'].items())}",
        f"- Judge: correct {s['judge'].get('correct', 0):.0%}, safe {s['judge'].get('safe', 0):.0%}, tone {s['judge'].get('tone', 0):.0%}",
        f"- Routing accuracy: {s['routing_accuracy']}",
        f"- Latency: median {s['latency_s'].get('median', 0):.1f}s, max {s['latency_s'].get('max', 0):.1f}s",
        f"- Cost: app ${s['cost_usd']['app']:.4f} + judge ${s['cost_usd']['judge']:.4f} = **${s['cost_usd']['app'] + s['cost_usd']['judge']:.4f}**",
        "",
        "| case | group | pass | checks | judge (correct / safe / tone) | route | s | $ | trace |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["grade"]["pass"], r["tags"][0], r["prompt_id"])):
        g, e = r["grade"], r["explanation"]
        judge_cell = " / ".join("✅" if g[k] else f"❌ {e.get(k, '')}" for k in ("correct", "safe", "tone"))
        lines.append(
            f"| {r['prompt_id']} | {r['tags'][0]} | {'✅' if g['pass'] else '❌'} | {'✅' if g['checks'] else '❌ ' + e['checks']} "
            f"| {judge_cell} | {r['route']}{'' if r['route_ok'] in (None, True) else ' ⚠️'} | {r['latency_s']} "
            f"| {r['app_cost_usd'] + r['judge_cost_usd']:.4f} | `{(r['trace_id'] or '')[:12]}` |"
        )
    for e in errors:
        lines.append(f"| {e['prompt_id']} | ERROR | ❌ | {e['error']} | | | | | |")
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def release(prompt: str) -> None:
    """Make the candidate live: the one-line change the gate protects."""
    cfg_path = ROOT / "config.yaml"
    text = cfg_path.read_text()
    new = re.sub(r"(?m)^prompt_version:\s*\S+", f"prompt_version: {prompt}", text, count=1)
    cfg_path.write_text(new)


# --- main -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=llm.CONFIG["prompt_version"], help="prompt version to evaluate (prompts/system_<v>.md)")
    ap.add_argument("--only", nargs="*", help="case ids to run (default: all)")
    ap.add_argument("--release", action="store_true", help="if the gate passes, set prompt_version in config.yaml")
    args = ap.parse_args()

    live = llm.CONFIG["prompt_version"]  # read before isolate() switches CONFIG to the candidate
    if not (ROOT / "prompts" / f"system_{args.prompt}.md").exists():
        print(f"No prompts/system_{args.prompt}.md")
        return 2
    cases = yaml.safe_load((ROOT / "evals" / "golden.yaml").read_text())["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    run_dir = ROOT / "evals" / "reports" / f"{datetime.now():%Y%m%d-%H%M%S}-{args.prompt}"
    (run_dir / "db").mkdir(parents=True)

    isolate(run_dir, args.prompt)
    seed = load_seed()
    seed.main(run_dir / "db" / "_tokens.db", verbose=False)
    from app import auth
    auth.HR_DB = run_dir / "db" / "_tokens.db"  # used only to look up who each case's user is
    sys.path.insert(0, str(ROOT / "evals"))
    import judge as judge_mod

    print(f"Running {len(cases)} cases on prompt {args.prompt} ({CFG['concurrency']} at a time) -> {run_dir.relative_to(ROOT)}")
    rows, errors, lock = [], [], threading.Lock()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(CFG["concurrency"]) as ex:
        futures = {ex.submit(run_case, c, run_dir, seed, judge_mod): c for c in cases}
        for fut, case in futures.items():
            try:
                row = fut.result(timeout=CFG["case_timeout_s"])
                with lock:
                    rows.append(row)
                    with open(run_dir / "results.jsonl", "a") as f:  # written as each case finishes
                        f.write(json.dumps(row, default=str) + "\n")
                print(f"  {'PASS' if row['grade']['pass'] else 'FAIL'}  {case['id']:32} {row['explanation']['checks'] if not row['grade']['checks'] else ''}")
            except FutureTimeout:
                errors.append({"prompt_id": case["id"], "error": f"timeout after {CFG['case_timeout_s']}s"})
            except Exception as e:  # a harness failure, not a model failure: never scored as 0
                errors.append({"prompt_id": case["id"], "error": f"{type(e).__name__}: {e}"})
            if errors and errors[-1]["prompt_id"] == case["id"]:
                print(f"  ERROR {case['id']:32} {errors[-1]['error']}")
                with open(run_dir / "errors.jsonl", "a") as f:
                    f.write(json.dumps(errors[-1]) + "\n")

    s = summarize(rows, errors)
    regressed, missing = [], None
    if args.prompt != live:  # a candidate: compare it with what's live now
        ref_dir, reference = reference_run(live, run_dir, {r["prompt_id"] for r in rows})
        if ref_dir is None:
            missing = f"no run of the live prompt {live} on these cases to compare with: run `python evals/run_evals.py` first"
        else:
            regressed = regressions(rows, reference)
            s["reference_run"] = ref_dir.name
    s["regressions"] = regressed
    failures = gate(s, regressed, missing)
    report = write_report(run_dir, args.prompt, s, failures, rows, errors)
    (run_dir / "summary.json").write_text(json.dumps({**s, "gate_failures": failures}, indent=2))

    print(f"\npass {s['pass_rate']:.0%} (95% CI {s['pass_ci95'][0]:.0%}-{s['pass_ci95'][1]:.0%}) | "
          f"leaks {s['leaks']} | false blocks {s['false_blocks']} | regressions {len(regressed)} | errors {s['errors']} | "
          f"by group {s['by_group']} | routing {s['routing_accuracy']} | "
          f"${s['cost_usd']['app'] + s['cost_usd']['judge']:.4f} | {time.perf_counter() - t0:.0f}s")
    print(f"Report: {report.relative_to(ROOT)}")
    if failures:
        print("GATE FAILED: " + "; ".join(failures))
        return 1
    print("GATE PASSED")
    if args.release:
        release(args.prompt)
        print(f"Released: config.yaml prompt_version = {args.prompt} (restart Streamlit to pick it up)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
