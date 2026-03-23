from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AuditFinding:
    severity: str  # "critical", "high", "medium", "low"
    category: str
    message: str
    file_path: str
    line_number: int | None = None


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)
    passed: bool = True
    scanned_files: int = 0

    def add(self, finding: AuditFinding) -> None:
        self.findings.append(finding)
        if finding.severity in ("critical", "high"):
            self.passed = False


# Patterns that must never appear in source code
_FORBIDDEN_PATTERNS: list[tuple[str, str, str]] = [
    (r"\bpickle\b", "critical", "pickle usage detected — RCE vector"),
    (
        r"\bpickle\.loads?\b",
        "critical",
        "pickle.load/loads — arbitrary code execution",
    ),
    (
        r"\bpickle\.dumps?\b",
        "critical",
        "pickle.dump/dumps — insecure serialization",
    ),
    (r"\bdill\b", "critical", "dill usage detected — same risk as pickle"),
    (r"\beval\s*\(", "critical", "eval() — arbitrary code execution"),
    (r"\bexec\s*\(", "critical", "exec() — arbitrary code execution"),
    (
        r"\b__import__\s*\(",
        "high",
        "__import__() — dynamic import, review needed",
    ),
    (r"\bos\.system\s*\(", "high", "os.system() — command injection risk"),
    (
        r"\bsubprocess\.call\s*\((?!.*check=True)",
        "medium",
        "subprocess.call without check=True",
    ),
]


def scan_source(
    src_dir: Path,
    exclude_patterns: list[str] | None = None,
) -> AuditReport:
    """Scan Python source files for forbidden patterns.

    Args:
        src_dir: root source directory (e.g., src/voxid/)
        exclude_patterns: glob patterns to exclude

    Returns:
        AuditReport with findings
    """
    report = AuditReport()
    excludes = exclude_patterns or []

    for py_file in src_dir.rglob("*.py"):
        skip = False
        for ex in excludes:
            if py_file.match(ex):
                skip = True
                break
        if skip:
            continue

        report.scanned_files += 1
        content = py_file.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue

            for pattern, severity, message in _FORBIDDEN_PATTERNS:
                if re.search(pattern, line):
                    if _is_in_string(line, pattern):
                        continue
                    report.add(
                        AuditFinding(
                            severity=severity,
                            category="code-pattern",
                            message=message,
                            file_path=str(py_file),
                            line_number=line_num,
                        )
                    )

    return report


def _is_in_string(line: str, pattern: str) -> bool:
    """Heuristic: check if a pattern match is inside a string literal."""
    match = re.search(pattern, line)
    if not match:
        return False
    pos = match.start()

    # Count unescaped quote characters before the match position
    before = line[:pos]
    single = before.count("'") - before.count("\\'")
    double = before.count('"') - before.count('\\"')
    return (single % 2 == 1) or (double % 2 == 1)


def scan_for_input_validation(
    src_dir: Path,
) -> AuditReport:
    """Check that API route handlers validate input.

    Looks for FastAPI route handlers and verifies they use
    Pydantic models for request bodies (not raw dict/Any).
    """
    report = AuditReport()

    routes_dir = src_dir / "api" / "routes"
    if not routes_dir.exists():
        return report

    for py_file in routes_dir.glob("*.py"):
        report.scanned_files += 1
        content = py_file.read_text(encoding="utf-8")

        if "dict[str, Any]" in content:
            lines = content.splitlines()
            for i, line in enumerate(lines, 1):
                if "dict[str, Any]" in line and "async def" in line:
                    report.add(
                        AuditFinding(
                            severity="medium",
                            category="input-validation",
                            message=(
                                "Route handler accepts raw "
                                "dict — use Pydantic model"
                            ),
                            file_path=str(py_file),
                            line_number=i,
                        )
                    )

    return report
