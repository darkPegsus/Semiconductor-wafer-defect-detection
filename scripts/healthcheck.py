#!/usr/bin/env python3
"""Startup health check script for WaferVision AI.

Runs model integrity verification, forward-pass smoke tests, and Grad-CAM
diagnostics to ensure the deployment environment is fully operational.
Exits with code 0 on success, code 1 on failure.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    CheckpointNotFoundError,
    HealthCheckError,
    ModelIntegrityError,
    get_app_config,
    run_startup_healthcheck,
)


def main() -> int:
    print("=" * 65)
    print(" WAFERVISION AI - STARTUP HEALTH CHECK & INTEGRITY AUDIT")
    print("=" * 65)

    config = get_app_config()
    print(f"Project Root:     {config.project_root}")
    print(f"Checkpoint Path:  {config.checkpoint_path}")
    print(f"Expected SHA-256: {config.expected_sha256}")
    print(f"Target Device:    {config.resolved_device} (setting: {config.device_setting})")
    print("-" * 65)

    try:
        report = run_startup_healthcheck(config)
        print(" HEALTH CHECK: PASSED (System Healthy)")
        print("-" * 65)
        print(json.dumps(report, indent=2))
        print("=" * 65)
        return 0

    except CheckpointNotFoundError as e:
        print(f"\n❌ CHECKPOINT NOT FOUND ERROR:\n{e}\n")
        return 1
    except ModelIntegrityError as e:
        print(f"\n❌ MODEL INTEGRITY ERROR:\n{e}\n")
        return 1
    except HealthCheckError as e:
        print(f"\n❌ HEALTH CHECK EXECUTION ERROR:\n{e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR DURING HEALTH CHECK:\n{e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
