#!/usr/bin/env python3
"""
Pre-commit hook for pyscn code quality check.
This script runs pyscn check to ensure code quality before commit.
Cross-platform compatible (Windows, Linux, macOS).
"""

import subprocess
import sys
import shutil
import io
from pathlib import Path

# Ensure stdout/stderr can handle UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Colors for output (ANSI codes)
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"  # No Color


def find_pyscn():
    """Find pyscn executable in PATH or common locations."""
    # Check if uv is available (preferred - uses project dependencies)
    uv_path = shutil.which("uv")
    if uv_path:
        return ["uv", "run", "pyscn"]
    
    # Check if pyscn is in PATH
    pyscn_path = shutil.which("pyscn")
    if pyscn_path:
        return ["pyscn"]
    
    # Check if uvx is available (can run pyscn without installation)
    uvx_path = shutil.which("uvx")
    if uvx_path:
        return ["uvx", "pyscn"]
    
    # Check if pipx is available (can install and run)
    pipx_path = shutil.which("pipx")
    if pipx_path:
        return ["pipx", "run", "pyscn"]
    
    return None


def run_pyscn_check():
    """Run pyscn check command on specific directories only."""
    pyscn_cmd = find_pyscn()
    
    if not pyscn_cmd:
        print(f"{YELLOW}Warning: pyscn not found. Install it with: uv sync --extra dev{NC}")
        print(f"{YELLOW}Skipping pyscn check for this commit.{NC}")
        return 0  # Don't fail commit if pyscn is not installed
    
    # Only check these directories: src/, tests/
    target_dirs = ["src", "tests"]
    
    print(f"{YELLOW}Running pyscn code quality check on: {', '.join(target_dirs)}...{NC}")
    print(f"Using: {' '.join(pyscn_cmd)}")
    
    try:
        # Run pyscn check on each directory
        all_passed = True
        for directory in target_dirs:
            dir_path = Path(directory)
            if not dir_path.exists():
                print(f"{YELLOW}Skipping {directory} (directory not found){NC}")
                continue
            
            print(f"\n{YELLOW}Checking {directory}...{NC}")
            result = subprocess.run(
                pyscn_cmd + ["check", directory],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=Path.cwd()
            )
            
            if result.returncode != 0:
                all_passed = False
                print(f"{RED}pyscn check failed for {directory}{NC}")
                if result.stdout:
                    try:
                        print(result.stdout)
                    except UnicodeEncodeError:
                        # Fallback: print without special characters
                        print(result.stdout.encode('ascii', errors='replace').decode('ascii'))
                if result.stderr:
                    try:
                        print(result.stderr)
                    except UnicodeEncodeError:
                        # Fallback: print without special characters
                        print(result.stderr.encode('ascii', errors='replace').decode('ascii'))
        
        if all_passed:
            print(f"\n{GREEN}pyscn check passed for all directories!{NC}")
            return 0
        else:
            print(f"\n{RED}pyscn check failed. Please fix code quality issues before committing.{NC}")
            print(f"{RED}Run 'pyscn analyze <directory>' for detailed report.{NC}")
            return 1
            
    except FileNotFoundError:
        print(f"{RED}Error: Could not execute pyscn command.{NC}")
        print(f"{YELLOW}Install pyscn with: pipx install pyscn{NC}")
        return 0  # Don't fail if command not found
    except Exception as e:
        print(f"{RED}Error running pyscn: {e}{NC}")
        return 1


if __name__ == "__main__":
    sys.exit(run_pyscn_check())

