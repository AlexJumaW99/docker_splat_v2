# docker_splat_v2 · GUI

A PySide6 desktop app that drives the `splat-ns-v2` container without touching
the CLI. It's a thin, read-mostly front end over `docker exec` — the same
`ns-process-data` / `ns-train` / `ns-export` / `ns-render` invocations you'd
run by hand, wrapped in a form with sensible defaults and per-project state.

## Run

```bash
./run.sh
```

First launch creates `.venv/` and installs PySide6, pydantic, and pillow-heif
(~90 MB). Subsequent launches are instant.

## What works today

| Tab       | State    | What it does                                                                                                                                 |
|-----------|----------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Dashboard | live     | Polls the Docker daemon, `splat-ns-v2` container, and GPU (via `nvidia-smi`). Shows the exact `docker build` / `docker create` commands if the container is missing, with copy buttons — never runs them for you. |
| Process   | live     | Form-driven `ns-process-data` (video or images). Per-project presets, advanced raw-flag passthrough, live ffprobe readout for video, live image count for image folders. Auto-converts HEIC/HEIF (iPhone) to max-quality JPEG with EXIF preserved; originals stashed in `heic_originals/`. |
| Train     | live     | Form-driven `ns-train`. Method picker (`splatfacto`, `splatfacto-big`, `splatfacto-mcmc`, `splatfacto-w`, `splatfacto-w-light`), iteration slider, viser + TensorBoard toggle, advanced raw-flag passthrough. Detects when the viser port becomes live and deep-links to `http://localhost:7007`. |
| Exports   | live     | Form-driven `ns-export gaussian-splat` (PLY) and `ns-render camera-path` (MP4 from viser keyframe JSON). Writes to `data/exports/splat/` and `data/renders/<project>/`; auto-suffixes on filename collisions so repeat exports never overwrite. |
| Viewers   | live     | Live port-status cards for viser (port 7007) and TensorBoard (port 6006). Run picker scoped to the active project — launch `ns-viewer` against a finished checkpoint, start/stop TensorBoard, and open either in the browser. Shows per-run TensorBoard-event availability. |
| Runs      | stub     | Placeholder for a training-run browser over `data/outputs/`.                                                                                 |

## Projects

A project bundles a dataset with its processing/training runs. Three ways to
create one (**+ New project**):

- **From video** — point at a video file inside `~/docker-stacks/docker_splat_v2/data/`.
  The project goes through Process → Train like the CLI workflow.
- **From images** — point at a folder of photos inside
  `~/docker-stacks/docker_splat_v2/data/input_images/`. Supports JPEG, PNG, TIFF,
  CR2, and HEIC/HEIF (auto-converted to JPEG). Goes Process → Train.
- **Import processed dataset** — point at an already-processed folder inside
  `~/docker-stacks/docker_splat_v2/data/`. The app auto-detects the layout:
  - `transforms.json` at the root → Nerfstudio dataparser
  - `sparse/0/cameras.bin` or `colmap/sparse/0/cameras.bin` → COLMAP dataparser,
    with the relative path recorded on the project so `ns-train` gets
    `--colmap-path <rel>` at train time.
  Imported projects skip straight to the Train tab.

Per-project state (presets, last-used params, dataparser metadata) persists
to `~/.config/docker_splat_v2_gui/projects.json`.

## Lineage & manifests

Process and Train runs write into the ULID-keyed warehouse at
`~/docker-stacks/docker_splat_v2/data_v2/` (bind-mounted into the container
at `/warehouse/`). Each run carries a `manifest.json` declaring its
upstream parent run, so a `.ckpt` traces back to its source video through
three file reads with no tooling required:

```
data_v2/sources/<source_ULID>/manifest.json     # stage=source, parent=null
data_v2/processed/<process_ULID>/manifest.json  # stage=process, parent.run_id=<source_ULID>
data_v2/trained/<train_ULID>/manifest.json      # stage=train,   parent.run_id=<process_ULID>
```

The schema (Pydantic v2 + JSON Schema Draft 2020-12, frozen at v1.0) lives
in [`app/manifest.py`](app/manifest.py) and the generated artifact at
[`app/manifest_schema.json`](app/manifest_schema.json). The side-effect
layer that mints ULIDs, computes checksums, copies sources, scrapes
metrics, and writes manifests is in [`app/warehouse.py`](app/warehouse.py).

### Why training also writes a `pending.json` sidecar

`ns-train` typically launches with `--viewer.quit-on-train-completion False`
so the viser viewer stays open after the last iteration. That keeps the
process pinned alive past the actual end of training, which means a
"manifest written on process exit" contract would never fire until the
user closes the viewer. Even worse: if the GUI dies in that window, the
successful run becomes an orphan on disk with no lineage breadcrumb.

The Train tab works around both problems:

1. **Write-ahead-log sidecar.** Before `ns-train` launches, the tab writes
   `data_v2/trained/<ULID>/pending.json` capturing the declared parent
   run, command, form params, and start time. The sidecar survives GUI
   death.
2. **`🎉 Training Finished 🎉` banner detection.** The tab watches
   `ns-train`'s stdout for nerfstudio's completion banner and finalizes
   the `TrainManifest` immediately when it appears — independent of
   process exit. The pending sidecar is then deleted, preserving the
   "`manifest.json` present = success" invariant.
3. **Process-exit fallback.** If the banner isn't detected (older
   nerfstudio, edited stdout, etc.) but the process eventually exits 0,
   the manifest is written from the exit path as a fallback.

### Recovering an orphan run

If you ever find `data_v2/trained/<ULID>/pending.json` with no
`manifest.json` (e.g. you killed the GUI while training was still alive),
reconstruct the manifest from `config.yml` + the pending sidecar:

```bash
./.venv/bin/python -m app.warehouse backfill-train ../data_v2/trained/<ULID>
```

The CLI writes a `manifest.json` with `provenance: "backfilled"` and
removes the pending sidecar. It works even without a pending sidecar
(falls back to scraping `config.yml` alone) and refuses to overwrite an
existing `manifest.json`.

## Safety guarantees

The Docker wrapper in `app/docker_service.py` exposes a narrow allow-list of
subcommands:

- Read-only: `docker version`, `docker info`, `docker inspect`,
  `docker image inspect`, `docker port`
- Lifecycle: `docker start`
- Exec: `docker exec` — used by two narrow helpers (in-container port probe
  and a `pkill`-based viser-port reclaim) plus `JobManager` for streaming
  long-running `ns-process-data` / `ns-train` / `ns-viewer` runs. The Viewers
  tab also uses detached `docker exec -d` to start TensorBoard.

It **cannot** build, create, remove, commit, tag, push, rename, stop, kill,
or run containers. Your `splat-nerfstudio-v2` image and `splat-ns-v2` container
are safe. If the container is missing, the Dashboard surfaces the `docker
create` command from `USER_GUIDE.md` with a copy button — it never runs it for
you.

## Config

State lives in `~/.config/docker_splat_v2_gui/`:

- `projects.json` — project list, active project, presets, last-used params.

Delete the directory to reset. The app recreates it on next launch.
