#!/bin/bash
# Forma for Linux. Run:  ./start_linux.sh   (or double-click if your file manager allows it)
# The first start installs a private Python and all libraries into ".venv" (5-10 minutes),
# and adds Forma to your applications menu.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
HERE="$(pwd)"
export PATH="$HOME/.local/bin:$PATH"
CHECK="import PySide6, pyvista, rhino3dm, OCP, trimesh, ezdxf, networkx"

# Qt 6 needs this system library on many distributions
if command -v ldconfig >/dev/null 2>&1 && ! ldconfig -p 2>/dev/null | grep -q libxcb-cursor; then
  echo "Note: Qt needs libxcb-cursor. If Forma does not open, install it:"
  echo "  Ubuntu/Debian:  sudo apt install libxcb-cursor0 libgl1"
  echo "  Fedora:         sudo dnf install xcb-util-cursor mesa-libGL"
  echo "  Arch:           sudo pacman -S xcb-util-cursor"
fi

if [ ! -x ".venv/bin/python" ] || ! .venv/bin/python -c "$CHECK" >/dev/null 2>&1 \
   || ! cmp -s requirements.txt .venv/.installed 2>/dev/null; then
  echo "== Setting up Forma (first start only) =="
  if ! command -v uv >/dev/null 2>&1; then
    echo "-- Installing the 'uv' Python manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || wget -qO- https://astral.sh/uv/install.sh | sh \
      || { echo "Could not download uv. Check your internet."; exit 1; }
    export PATH="$HOME/.local/bin:$PATH"
  fi
  [ -x ".venv/bin/python" ] || uv venv --python 3.12 .venv || exit 1
  echo "-- Installing libraries (Qt, VTK, OpenCascade...). About 1 GB the first time."
  uv pip install --python .venv/bin/python -r requirements.txt || { echo "Install failed."; exit 1; }
  cp requirements.txt .venv/.installed
  # applications-menu entry
  mkdir -p "$HOME/.local/share/applications"
  cat > "$HOME/.local/share/applications/forma.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Forma
Comment=NURBS CAD: drafting, 3D modeling, components and nodes
Exec="$HERE/start_linux.sh" %f
Icon=$HERE/forma/resources/forma_icon.png
Terminal=false
Categories=Graphics;3DGraphics;Engineering;
DESK
  echo "Added Forma to your applications menu."
fi

exec .venv/bin/python -m forma "$@"
