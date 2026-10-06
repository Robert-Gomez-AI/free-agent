"""Offline tests for the mode tool packs (no network)."""
from __future__ import annotations

from pathlib import Path

import pytest

from free_agent.tools.domains import code, finance, science, security, set_writable_root


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    set_writable_root(None)
    yield tmp_path
    set_writable_root(None)


# ─── code ───────────────────────────────────────────────────────────────────


def test_read_tools(repo: Path) -> None:
    assert "Python (pyproject)" in code.project_overview.invoke({})
    assert "pkg/mod.py" in code.glob_files.invoke({"pattern": "**/*.py"})
    assert "mod.py:1" in code.grep_code.invoke({"pattern": "def hello"})
    assert "1\tdef hello():" in code.view_file.invoke({"path": "pkg/mod.py"})


def test_writes_refused_without_writable(repo: Path) -> None:
    out = code.str_replace.invoke({"path": "pkg/mod.py", "old_string": "hi", "new_string": "yo"})
    assert "/writable on" in out
    assert "'hi'" in (repo / "pkg" / "mod.py").read_text()


def test_str_replace_and_create(repo: Path) -> None:
    set_writable_root(repo)
    assert "edited" in code.str_replace.invoke({"path": "pkg/mod.py", "old_string": "'hi'", "new_string": "'yo'"})
    assert "'yo'" in (repo / "pkg" / "mod.py").read_text()
    assert "not found" in code.str_replace.invoke({"path": "pkg/mod.py", "old_string": "zzz", "new_string": "q"})
    assert "wrote" in code.create_file.invoke({"path": "new/a.txt", "content": "x\n"})
    assert "already exists" in code.create_file.invoke({"path": "new/a.txt", "content": "y"})


def test_str_replace_requires_unique(repo: Path) -> None:
    set_writable_root(repo)
    (repo / "dup.txt").write_text("a a", encoding="utf-8")
    assert "matches 2 times" in code.str_replace.invoke({"path": "dup.txt", "old_string": "a", "new_string": "b"})
    code.str_replace.invoke({"path": "dup.txt", "old_string": "a", "new_string": "b", "replace_all": True})
    assert (repo / "dup.txt").read_text() == "b b"


def test_writes_cannot_escape_root(repo: Path) -> None:
    set_writable_root(repo / "pkg")
    assert "outside the writable root" in code.create_file.invoke({"path": "../escape.txt", "content": "x"})
    assert not (repo / "escape.txt").exists()


def test_git_inspect_blocks_mutations() -> None:
    assert "not allowed" in code.git_inspect.invoke({"subcommand": "push"})


# ─── science ────────────────────────────────────────────────────────────────


def test_stats_summary() -> None:
    out = science.stats_summary.invoke({"values": "1, 2, 3, 4, 5"})
    assert "mean=3" in out and "median=3" in out


def test_linear_fit_exact() -> None:
    out = science.linear_fit.invoke({"x_values": "1 2 3 4", "y_values": "3 5 7 9"})
    assert "y = 1 + 2·x" in out and "R²=1.00000" in out


def test_python_compute() -> None:
    assert "42" in science.python_compute.invoke({"code": "print(6*7)"})


# ─── security ───────────────────────────────────────────────────────────────


def test_hash_and_codec() -> None:
    assert "sha256: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" in security.hash_data.invoke({"text": "hello"})
    assert security.codec_transform.invoke({"data": "aGVsbG8=", "operation": "base64_decode"}) == "hello"
    assert security.codec_transform.invoke({"data": "a b", "operation": "url_encode"}) == "a%20b"


def test_jwt_alg_none_flagged() -> None:
    assert "CRITICAL: alg=none" in security.jwt_inspect.invoke({"token": "eyJhbGciOiJub25lIn0.eyJzdWIiOiIxIn0."})


def test_extract_iocs_defanged() -> None:
    out = security.extract_iocs.invoke({"text": "c2 hxxp://evil[.]com from 10.1.2.3 CVE-2024-3094"})
    assert "10.1.2.3" in out and "evil[.]com" in out and "CVE-2024-3094" in out


def test_cve_id_validation() -> None:
    assert "not a valid CVE" in security.cve_lookup.invoke({"cve_id": "log4shell"})


# ─── finance ────────────────────────────────────────────────────────────────


def test_cash_flow_analysis() -> None:
    out = finance.cash_flow_analysis.invoke({"cash_flows": "-1000, 300, 400, 500, 200", "discount_rate_pct": 10})
    assert "NPV @ 10.0% = 115.57" in out and "IRR = 15.3" in out and "ACCEPT" in out


def test_loan_schedule_zero_balance() -> None:
    out = finance.loan_schedule.invoke({"principal": 12000, "annual_rate_pct": 0, "years": 1})
    assert "payment = 1,000.00" in out and "total interest = 0.00" in out


def test_time_value_of_money() -> None:
    out = finance.time_value_of_money.invoke(
        {"present_value": 1000, "annual_rate_pct": 10, "years": 1, "periods_per_year": 1}
    )
    assert "FV = 1,100.00" in out


def test_portfolio_risk_drawdown() -> None:
    out = finance.portfolio_risk.invoke({"prices": "100 120 90 95 110"})
    assert "max drawdown=-25.00%" in out
