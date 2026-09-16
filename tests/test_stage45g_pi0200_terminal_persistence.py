"""CPU-only regression for PI0200 terminal evidence ordering."""

import ast
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "scripts" / "stage43_corelab_initial_condition.py"


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if not isinstance(node.func, ast.Attribute):
        return None
    if isinstance(node.func.value, ast.Name):
        return f"{node.func.value.id}.{node.func.attr}"
    return None


def test_terminal_segments_are_buffered_before_fail_closed_checks() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    run = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_run")
    checked_sites = []
    for node in ast.walk(run):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        calls = [_call_name(statement.value) for statement in body if isinstance(statement, ast.Expr)]
        if "rows_before.append" in calls or "rows_after.append" in calls:
            checked_sites.append(calls)
    assert len(checked_sites) == 2
    for calls in checked_sites:
        check = calls.index("_check_rows")
        assert any(name in calls[:check] for name in ("rows_before.append", "rows_after.append"))
        assert "counter_events.append" in calls[:check]
