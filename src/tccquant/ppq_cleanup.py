from __future__ import annotations

from pathlib import Path
import shutil

DEFAULT_DELETE_PATTERNS = [
    "tests",
    "docs",
    "samples",
    "docker",
    ".github",
    "*.md",
]


def prune_ppq_tree(ppq_root: str | Path, delete_patterns: list[str] | None = None, dry_run: bool = False) -> list[str]:
    """Delete unnecessary files from a PPQ tree while keeping core python package.

    Args:
        ppq_root: Root of existing PPQ checkout.
        delete_patterns: Glob patterns to delete.
        dry_run: Only return candidate paths without deletion.
    """
    root = Path(ppq_root)
    if not root.exists():
        raise FileNotFoundError(root)

    patterns = delete_patterns or DEFAULT_DELETE_PATTERNS
    removed: list[str] = []
    for pattern in patterns:
        for target in root.glob(pattern):
            if target.name in {"ppq", "setup.py", "pyproject.toml"}:
                continue
            removed.append(str(target))
            if dry_run:
                continue
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
    return removed
