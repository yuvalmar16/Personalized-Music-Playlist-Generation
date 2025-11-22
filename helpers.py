import sys
import subprocess

def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])