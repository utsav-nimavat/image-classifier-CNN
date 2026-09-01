"""Vercel serverless entrypoint."""
import sys
from pathlib import Path

# the function is executed from api/, but app.py and predict.py sit one level up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402  (must come after the sys.path line above)
