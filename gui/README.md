# docker_splat · GUI

A PySide6 desktop app for the docker_splat pipeline.

## Run

```bash
./run.sh
```

First launch creates `.venv/` and installs PySide6 + pydantic (~80 MB).
Subsequent launches are instant.

## What it does (v0.1)

- **Dashboard** — live status of the Docker daemon, the `splat-ns` container,
  and your GPU (via `nvidia-smi`).
- Process / Train / Viewers / Runs panels are stubs — wiring them up next.

## Safety guarantees

This app is a **read-mostly** Docker client. The wrapper in
`app/docker_service.py` exposes only these subcommands:

- `docker version`, `docker info`
- `docker inspect`, `docker image inspect`, `docker port`
- `docker start`, `docker exec`

It **cannot** build, create, remove, commit, tag, push, or run containers.
Your existing `splat-nerfstudio` image and `splat-ns` container are safe.

If the container is missing, the GUI shows the create command from
`USER_GUIDE.md` with a copy button — it never runs it for you.

## Config

State lives in `~/.config/docker_splat_gui/`.
