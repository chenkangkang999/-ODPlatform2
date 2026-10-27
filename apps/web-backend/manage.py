"""Local Django entry point; run from any working directory."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps/platform/src"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fishweb.settings")

if __name__ == "__main__":
    from fishweb.env import load_local_env

    load_local_env()
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
