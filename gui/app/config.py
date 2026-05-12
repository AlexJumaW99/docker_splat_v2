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

# Manifest-aware warehouse root. Bind-mounted into the splat-ns-v2 container
# at CONTAINER_WAREHOUSE (see docker-compose.yml). Every new artifact written
# after the warehouse migration lives under here, ULID-keyed:
#   data_v2/sources/<ULID>/{<filename>, manifest.json}
#   data_v2/processed/<ULID>/{transforms.json, colmap/, images/, manifest.json}
#   data_v2/trained/<ULID>/   data_v2/exports/<ULID>/   data_v2/renders/<ULID>/
WAREHOUSE_DIR = REPO_ROOT / "data_v2"
CONTAINER_WAREHOUSE = "/warehouse"
# Mount-point inside the container for the legacy data tree. Kept as a
# constant so warehouse path-translation helpers can mirror the bind in
# docker-compose.yml without re-encoding the literal in every caller.
CONTAINER_WORKSPACE = "/workspace"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
