#!/usr/bin/env python3
"""Launcher for the smart notebook generator skill assets.

This wrapper lets the skill live in VS Code's recognized `.github/skills/`
directory while reusing the main implementation stored at the repository root.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    root_script = (
        Path(__file__).resolve().parents[4]
        / "smart-notebook-generator"
        / "scripts"
        / "create_smart_notebook.py"
    )
    runpy.run_path(str(root_script), run_name="__main__")