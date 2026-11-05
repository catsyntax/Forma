#!/bin/bash
# Forma for macOS. Double-click to start.
# The first start installs a private Python and all libraries into the ".venv" folder here (5-10 minutes).
# Later starts open Forma in a few seconds. Nothing is installed system-wide.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
CHECK="import PySide6, pyvista, rhino3dm, OCP, trimesh, ezdxf, networkx"

if [ ! -x ".venv/bin/python" ] || ! .venv/bin/python -c "$CHECK" >/dev/null 2>&1 \
   || ! cmp -s requirements.txt .venv/.installed 2>/dev/null; then
  echo "== Setting up Forma (first start only) =="
  if ! command -v uv >/dev/null 2>&1; then
    echo "-- Installing the 'uv' Python manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Could not download uv. Check your internet."; read -r; exit 1; }
    export PATH="$HOME/.local/bin:$PATH"
  fi
  [ -x ".venv/bin/python" ] || uv venv --python 3.12 .venv || { read -r; exit 1; }
  echo "-- Installing libraries (Qt, VTK, OpenCascade...). About 1 GB the first time."
  uv pip install --python .venv/bin/python -r requirements.txt || { echo "Install failed."; read -r; exit 1; }
  cp requirements.txt .venv/.installed
fi

echo "Starting Forma... (keep this window open while you work)"
exec .venv/bin/python -m forma "$@"
