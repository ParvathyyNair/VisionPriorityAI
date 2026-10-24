"""The dashboard and risk layer must work with the network disabled."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_offline_quick_check_passes():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "offline_check.py"), "--quick"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "OFFLINE READY" in r.stdout
