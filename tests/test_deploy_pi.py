"""tools/deploy_pi.sh: help works and a missing host is an error (no SSH needed)."""

from __future__ import annotations

import os
import subprocess

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "tools", "deploy_pi.sh")


def test_help() -> None:
    out = subprocess.run(["bash", SCRIPT, "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "--pull" in out.stdout


def test_missing_host_fails() -> None:
    out = subprocess.run(["bash", SCRIPT], capture_output=True, text=True)
    assert out.returncode == 2 and "missing user@host" in out.stderr
