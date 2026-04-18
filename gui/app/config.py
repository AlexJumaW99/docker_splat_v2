from pathlib import Path

APP_NAME = "docker_splat"

CONTAINER_NAME = "splat-ns"
IMAGE_NAME = "splat-nerfstudio"

VISER_PORT = 7007
TENSORBOARD_PORT = 6006

CONFIG_DIR = Path.home() / ".config" / "docker_splat_gui"
PROJECTS_FILE = CONFIG_DIR / "projects.json"

REPO_ROOT = Path.home() / "Desktop" / "docker_splat"
DATA_DIR = REPO_ROOT / "data"
USER_GUIDE = REPO_ROOT / "USER_GUIDE.md"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
