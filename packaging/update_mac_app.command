#!/bin/bash
# Installs any new libraries into Forma.app's Python environment, then opens Forma.
cd "$(dirname "$0")/.."
SRC="$(pwd)"
APP="${1:-$SRC/../Forma.app}"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
echo "Updating $APP ..."
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv pip install --python "$APP/Contents/Resources/venv/bin/python" -r "$SRC/requirements.txt" 2>&1 | tee "$SRC/update.log"
echo "Done. Opening Forma..."
open "$APP"
sleep 2
