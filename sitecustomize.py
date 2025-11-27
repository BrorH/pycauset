"""Ensure the local python/ directory is on sys.path for in-repo tooling."""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent
_python_dir = _repo_root / "python"
if _python_dir.exists():
    path_str = str(_python_dir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
