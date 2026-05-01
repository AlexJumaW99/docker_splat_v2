# docker_splat_v2 · GUI

A PySide6 desktop app that drives the `splat-ns-v2` container without touching
the CLI. It's a thin, read-mostly front end over `docker exec` — the same
`ns-process-data` / `ns-train` / `ns-export` / `ns-render` invocations you'd
run by hand, wrapped in a form with sensible defaults and per-project state.

## Run

```bash
./run.sh
```

First launch creates `.venv/` and installs PySide6 + pydantic (~80 MB).
Subsequent launches are instant.

## What works today

| Tab       | State    | What it does                                                                                                                                 |
|-----------|----------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Dashboard | live     | Polls the Docker daemon, `splat-ns-v2` container, and GPU (via `nvidia-smi`). Shows the exact `docker build` / `docker create` commands if the container is missing, with copy buttons — never runs them for you. |
| Process   | live     | Form-driven `ns-process-data video`. Per-project presets, advanced raw-flag passthrough, live ffprobe readout of the source video.           |
| Train     | live     | Form-driven `ns-train`. Method picker (`splatfacto`, `splatfacto-big`, `splatfacto-mcmc`, `splatfacto-w`, `splatfacto-w-light`), iteration slider, viser + TensorBoard toggle, advanced raw-flag passthrough. Detects when the viser port becomes live and deep-links to `http://localhost:7007`. |
| Exports   | live     | Form-driven `ns-export gaussian-splat` (PLY) and `ns-render camera-path` (MP4 from viser keyframe JSON). Writes to `data/exports/splat/` and `data/renders/<project>/`; auto-suffixes on filename collisions so repeat exports never overwrite. |
| Viewers   | stub     | Placeholder — viser + TensorBoard are reachable directly in the browser in the meantime.                                                     |
| Runs      | stub     | Placeholder for a training-run browser over `data/outputs/`.                                                                                 |

## Projects

A project bundles a dataset with its processing/training runs. Two ways to
create one (**+ New project**):

- **From video** — point at a video file inside `~/docker-stacks/docker_splat_v2/data/`.
  The project goes through Process → Train like the CLI workflow.
- **Import processed dataset** — point at an already-processed folder inside
  `~/docker-stacks/docker_splat_v2/data/`. The app auto-detects the layout:
  - `transforms.json` at the root → Nerfstudio dataparser
  - `sparse/0/cameras.bin` or `colmap/sparse/0/cameras.bin` → COLMAP dataparser,
    with the relative path recorded on the project so `ns-train` gets
    `--colmap-path <rel>` at train time.
  Imported projects skip straight to the Train tab.

Per-project state (presets, last-used params, dataparser metadata) persists
to `~/.config/docker_splat_v2_gui/projects.json`.

## Safety guarantees

The Docker wrapper in `app/docker_service.py` exposes a narrow allow-list of
subcommands:

- Read-only: `docker version`, `docker info`, `docker inspect`,
  `docker image inspect`, `docker port`
- Lifecycle: `docker start`
- Exec: `docker exec` — used by two narrow helpers (in-container port probe
  and a `pkill`-based viser-port reclaim) plus `JobManager` for streaming
  long-running `ns-process-data` / `ns-train` runs.

It **cannot** build, create, remove, commit, tag, push, rename, stop, kill,
or run containers. Your `splat-nerfstudio-v2` image and `splat-ns-v2` container
are safe. If the container is missing, the Dashboard surfaces the `docker
create` command from `USER_GUIDE.md` with a copy button — it never runs it for
you.

## Config

State lives in `~/.config/docker_splat_v2_gui/`:

- `projects.json` — project list, active project, presets, last-used params.

Delete the directory to reset. The app recreates it on next launch.
