"""Run the existing BNCI15 shell script, then the BNCI14 shell script.

The shell scripts keep their own experiment configuration.  This file only
coordinates their order and therefore requires Bash (Git Bash is sufficient
on Windows).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import winreg
except ImportError:  # pragma: no cover - only used on Windows
    winreg = None


EXPERIMENTS_DIR = Path(__file__).resolve().parent


def find_bash() -> Optional[str]:
    """Find Bash in PATH or in the usual Git for Windows locations."""
    bash = shutil.which("bash") or shutil.which("bash.exe")
    if bash:
        return bash

    candidates = [
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/usr/bin/bash.exe",
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Git/bin/bash.exe",
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Git/usr/bin/bash.exe",
    ]

    # Git may be installed on a non-system drive (for example D:\\Git).
    git = shutil.which("git") or shutil.which("git.exe")
    if git:
        git_root = Path(git).resolve().parent.parent
        candidates.extend((git_root / "bin/bash.exe", git_root / "usr/bin/bash.exe"))

    if winreg is not None:
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for key_name in (
                r"SOFTWARE\GitForWindows",
                r"SOFTWARE\WOW6432Node\GitForWindows",
            ):
                try:
                    with winreg.OpenKey(hive, key_name) as key:
                        install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                    root = Path(install_path)
                    candidates.extend((root / "bin/bash.exe", root / "usr/bin/bash.exe"))
                except (FileNotFoundError, OSError):
                    pass

    return next((str(path) for path in candidates if path.is_file()), None)


def run_script(bash: str, script_name: str) -> None:
    script = EXPERIMENTS_DIR / script_name
    if not script.is_file():
        raise FileNotFoundError(f"Missing script: {script}")

    print(f"Starting {script_name}...", flush=True)
    subprocess.run([bash, str(script)], cwd=EXPERIMENTS_DIR, check=True)
    print(f"{script_name} completed.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bash",
        default=None,
        help="Path to bash executable; otherwise PATH/Git Bash locations are searched",
    )
    args = parser.parse_args()

    bash = args.bash or find_bash()
    if not bash:
        print(
            "Error: Bash was not found. Install Git for Windows or pass --bash PATH.",
            file=sys.stderr,
        )
        return 127

    try:
        # run_script(bash, "script_bnci15.sh")
        run_script(bash, "script_bnci14.sh")
    except subprocess.CalledProcessError as exc:
        print(
            f"Stopped: {exc.cmd[-1] if isinstance(exc.cmd, list) else exc.cmd} "
            f"exited with code {exc.returncode}.",
            file=sys.stderr,
        )
        return exc.returncode or 1
    except (FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
