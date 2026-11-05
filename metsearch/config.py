"""Settings, read once from the environment (and .env if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# Department IDs from GET /departments. The first pass uses just one.
DEPARTMENT_NAMES = {
    6: "Asian Art",
    9: "Drawings and Prints",
    10: "Egyptian Art",
    11: "European Paintings",
}


def _int_list(value: str) -> list[int]:
    return [int(v) for v in value.replace(" ", "").split(",") if v]


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL", "postgresql://met:met@localhost:5432/met"))
    met_api_base: str = field(default_factory=lambda: os.getenv(
        "MET_API_BASE", "https://collectionapi.metmuseum.org/public/collection/v1"))
    # The Met asks for well under 80 req/s. Stay far below that.
    requests_per_second: float = field(default_factory=lambda: float(os.getenv("MET_RPS", "6")))
    image_requests_per_second: float = field(default_factory=lambda: float(os.getenv("IMAGE_RPS", "5")))
    departments: list[int] = field(default_factory=lambda: _int_list(os.getenv("DEPARTMENTS", "6")))
    # Hard cap on objects per department. Scope is the main risk: keep this small
    # until the whole pipeline works end to end.
    max_objects_per_department: int = field(default_factory=lambda: int(os.getenv("MAX_OBJECTS_PER_DEPT", "5000")))
    image_dir: Path = field(default_factory=lambda: Path(os.getenv("IMAGE_DIR", "data/images")))
    clip_model: str = field(default_factory=lambda: os.getenv("CLIP_MODEL", "ViT-B-32"))
    clip_pretrained: str = field(default_factory=lambda: os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k"))
    # "clip" for real use; "fake" is a deterministic stand-in for tests/offline dev.
    embedder: str = field(default_factory=lambda: os.getenv("EMBEDDER", "clip"))
    user_agent: str = field(default_factory=lambda: os.getenv(
        "USER_AGENT", "metvision/0.1 (portfolio project; polite crawler)"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"))


def get_settings() -> Settings:
    return Settings()
