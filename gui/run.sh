#!/usr/bin/env bash
# Launcher: creates a venv on first run, installs deps, starts the GUI.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

if [ ! -d .venv ]; then
    echo "[gui] creating virtualenv at .venv"
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip wheel >/dev/null
    .venv/bin/pip install -e .
fi

# Wayland hints — the app is tested on Wayland but falls back cleanly to xcb.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland;xcb}"

exec .venv/bin/python -m app "$@"
