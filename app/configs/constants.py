from pathlib import Path
from typing import Final

APP_ID: Final[str] = ""
APP_NAME: Final[str] = "lawer-be"

ROOT_DIR: Final[Path] = Path(__file__).parent.parent.parent
LOGGING_CONFIG_PATH: Final[Path] = ROOT_DIR / "logging.yaml"
