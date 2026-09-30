"""Load config.yaml and resolve absolute paths."""
from pathlib import Path
import yaml


class Config:
    def __init__(self, data: dict, root: Path):
        self._data = data
        self.root = root

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def path(self, key: str) -> Path:
        """Resolve a paths.<key> entry to an absolute Path."""
        rel = self._data["paths"][key]
        return self.root / rel

    @property
    def raw(self) -> dict:
        return self._data


def load_config(config_path: str = "config.yaml") -> Config:
    path = Path(config_path).resolve()
    root = path.parent
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data, root)
