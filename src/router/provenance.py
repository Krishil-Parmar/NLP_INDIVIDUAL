"""Provenance for result files (rigour rule 5).

Every result file records the git commit, config, seed, dataset revision and
timestamp, so any number in the report can be traced to the code that made it.
"""

import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Packages whose versions are worth recording next to a result.
_TRACKED_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "sentence-transformers",
    "scikit-learn",
    "pandas",
    "numpy",
]


def _git(*args: str) -> str:
    """Run a git command in the project root and return its output ("" if git fails)."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_commit() -> str:
    """Current commit hash, or "unknown" if there is no commit yet."""
    return _git("rev-parse", "HEAD") or "unknown"


def git_is_dirty() -> bool:
    """True if there are uncommitted changes to tracked or new files."""
    return _git("status", "--porcelain") != ""


def package_versions() -> dict[str, str]:
    """Installed version of each tracked package."""
    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def run_metadata(cfg: dict, seed: int | None = None) -> dict:
    """Everything a result file must record about the run that produced it."""
    revisions = {
        key: value["revision"]
        for key, value in cfg.items()
        if isinstance(value, dict) and "revision" in value
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "seed": seed,
        "dataset_revisions": revisions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "config": cfg,
    }


def metadata_markdown(meta: dict) -> str:
    """Format run_metadata output as a short markdown block for the top of a report."""
    lines = [
        f"- Generated (UTC): {meta['timestamp_utc']}",
        f"- Git commit: `{meta['git_commit']}`"
        + (" (uncommitted changes present)" if meta["git_dirty"] else ""),
        f"- Seed: {meta['seed']}",
        f"- Python {meta['python']} on {meta['platform']}",
    ]
    for name, rev in meta["dataset_revisions"].items():
        lines.append(f"- Dataset `{name}` revision: `{rev}`")
    packages = ", ".join(f"{k} {v}" for k, v in meta["packages"].items())
    lines.append(f"- Packages: {packages}")
    return "\n".join(lines)
