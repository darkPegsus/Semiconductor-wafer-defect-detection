"""Convenience runner script for the WaferVision AI Streamlit application."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    """Launch the Streamlit web application using current Python environment."""
    app_path = PROJECT_ROOT / "app" / "main.py"
    if not app_path.exists():
        print(f"Error: Application entry point not found at {app_path}")
        sys.exit(1)

    print("=================================================================")
    print("   LAUNCHING WAFERVISION AI (STREAMLIT DASHBOARD)                ")
    print("=================================================================")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"App Entry:    {app_path}")
    print("Default URL:  http://localhost:8501")
    print("=================================================================\n")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--theme.base=dark",
        "--theme.primaryColor=#00d4ff",
        "--theme.backgroundColor=#0b0f19",
        "--theme.secondaryBackgroundColor=#121826",
        "--theme.textColor=#f1f5f9",
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nWaferVision AI server stopped gracefully.")


if __name__ == "__main__":
    main()
