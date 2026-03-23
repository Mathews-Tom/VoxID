from __future__ import annotations

from pathlib import Path

from voxid.security.audit import (
    AuditFinding,
    AuditReport,
    scan_for_input_validation,
    scan_source,
)


def _write_py(tmp_path: Path, name: str, content: str) -> Path:
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


def test_scan_source_no_findings() -> None:
    src_dir = Path(__file__).parent.parent / "src" / "voxid"
    report = scan_source(src_dir)
    assert report.passed is True


def test_scan_source_detects_pickle(tmp_path: Path) -> None:
    _write_py(tmp_path, "bad.py", "import pickle\n")
    report = scan_source(tmp_path)
    assert any(f.severity == "critical" for f in report.findings)


def test_scan_source_detects_eval(tmp_path: Path) -> None:
    _write_py(tmp_path, "bad_eval.py", "result = eval(input())\n")
    report = scan_source(tmp_path)
    assert any(f.severity == "critical" for f in report.findings)


def test_scan_source_detects_exec(tmp_path: Path) -> None:
    _write_py(tmp_path, "bad_exec.py", "exec(code)\n")
    report = scan_source(tmp_path)
    assert any(f.severity == "critical" for f in report.findings)


def test_scan_source_skips_comments(tmp_path: Path) -> None:
    _write_py(tmp_path, "comments.py", "# pickle is bad, never use it\n")
    report = scan_source(tmp_path)
    assert report.findings == []


def test_scan_source_skips_string_literals(tmp_path: Path) -> None:
    _write_py(tmp_path, "strings.py", 'msg = "pickle is not used here"\n')
    report = scan_source(tmp_path)
    assert report.findings == []


def test_scan_source_counts_files(tmp_path: Path) -> None:
    _write_py(tmp_path, "a.py", "x = 1\n")
    _write_py(tmp_path, "b.py", "y = 2\n")
    report = scan_source(tmp_path)
    assert report.scanned_files > 0


def test_scan_for_input_validation_clean() -> None:
    src_dir = Path(__file__).parent.parent / "src" / "voxid"
    report = scan_for_input_validation(src_dir)
    critical_or_high = [
        f for f in report.findings if f.severity in ("critical", "high")
    ]
    assert critical_or_high == []


def test_no_pickle_eval_exec_in_codebase() -> None:
    src_dir = Path(__file__).parent.parent / "src" / "voxid"
    report = scan_source(src_dir)
    assert report.passed is True
    critical = [f for f in report.findings if f.severity == "critical"]
    assert critical == []


def test_audit_report_passed_flag() -> None:
    report = AuditReport()
    assert report.passed is True

    report.add(
        AuditFinding(
            severity="critical",
            category="code-pattern",
            message="test critical finding",
            file_path="fake.py",
            line_number=1,
        )
    )
    assert report.passed is False
