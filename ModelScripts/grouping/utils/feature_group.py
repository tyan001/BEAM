import json
from pathlib import Path

_JSON_PATH = Path(__file__).parent / "feature_groups.json"


def load_feature_groups() -> dict[str, list[str]]:
    with open(_JSON_PATH) as f:
        return json.load(f)
