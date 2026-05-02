from pathlib import Path

APP_NAME = "docker_splat_v2"

CONTAINER_NAME = "splat-ns-v2"
IMAGE_NAME = "splat-nerfstudio-v2"

VISER_PORT = 7007
TENSORBOARD_PORT = 6006

VIZER_CONTAINER = "viser-colmap"
VIZER_VIEWER_PORT = 7070

CONFIG_DIR = Path.home() / ".config" / "docker_splat_v2_gui"
PROJECTS_FILE = CONFIG_DIR / "projects.json"

REPO_ROOT = Path.home() / "docker-stacks" / "docker_splat_v2"
DATA_DIR = REPO_ROOT / "data"
USER_GUIDE = REPO_ROOT / "USER_GUIDE.md"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
