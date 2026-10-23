"""Read optional dataset metadata files."""

import json
from pathlib import Path

def load_metadata(root, filename, default):
    path = Path(root) / filename
    if not path.is_file():
        return default
    with path.open() as stream:
        return json.load(stream)
