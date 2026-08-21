"""Persist shared workshop state between independently executed notebooks."""

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = ROOT / "validation" / "aws_state.json"


def state_path():
    return Path(os.environ.get("RAG_WORKSHOP_STATE_PATH", DEFAULT_STATE_PATH))


def persist_workshop_state(state):
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return state


def load_workshop_state():
    path = state_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
