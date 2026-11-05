#!/bin/bash
# Builds Forma.app next to this project folder (double-click, or: bash packaging/make_mac_app.command [destination]).
# The app contains Forma's code and its own Python environment, so it runs without Terminal.
set -e
cd "$(dirname "$0")/.."
SRC="$(pwd)"
DEST="${1:-$SRC/..}"
APP="$DEST/Forma.app"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
echo "Building $APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/app"

# --- code
rm -rf "$APP/Contents/Resources/app"
mkdir -p "$APP/Contents/Resources/app"
cp -R "$SRC/forma" "$SRC/main.py" "$SRC/packaging" "$SRC/examples" "$SRC/requirements.txt" "$SRC/README.md" "$APP/Contents/Resources/app/"
find "$APP/Contents/Resources/app" -name "__pycache__" -type d -prune -exec rm -rf {} +
cp "$SRC/packaging/Forma.icns" "$APP/Contents/Resources/Forma.icns"

# --- Python environment (reuse the project's .venv if present, otherwise create one)
VENV="$APP/Contents/Resources/venv"
if [ ! -e "$VENV/pyvenv.cfg" ]; then
  if [ -e "$SRC/.venv/pyvenv.cfg" ]; then
    echo "Moving the existing Python environment into the app..."
    mv "$SRC/.venv" "$VENV"
  else
    echo "Creating a Python 3.12 environment inside the app (a few minutes)..."
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" -r "$SRC/requirements.txt"
  fi
fi

# --- launcher
cat > "$APP/Contents/MacOS/Forma" <<'LAUNCH'
#!/bin/bash
RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
LOGDIR="$HOME/Library/Logs/Forma"
mkdir -p "$LOGDIR"
cd "$RES/app" || exit 1
if [ ! -x "$RES/venv/bin/python" ]; then
  osascript -e 'display alert "Forma" message "The Python environment inside Forma.app is missing. Run packaging/make_mac_app.command again."'
  exit 1
fi
# install new libraries after an update (one time per change of requirements.txt)
REQ="$RES/app/requirements.txt"
STAMP="$RES/venv/.requirements_installed"
if [ -f "$REQ" ] && ! cmp -s "$REQ" "$STAMP"; then
  osascript -e 'display notification "Installing new components (one time, about a minute)..." with title "Forma"'
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  PY="$RES/venv/bin/python"
  if "$UV" pip install --python "$PY" -r "$REQ" >> "$LOGDIR/update.log" 2>&1; then
    cp "$REQ" "$STAMP"
  fi
  # Slim down: Forma only needs the core Qt modules, not the web engine, PDF, 3D/QML add-ons.
  if "$PY" -c "import importlib.metadata as m; m.version('pyside6-addons')" >/dev/null 2>&1; then
    osascript -e 'display notification "Removing unused libraries to make Forma smaller..." with title "Forma"'
    "$UV" pip uninstall --python "$PY" pyside6 pyside6-addons pyside6-webengine pyside6-pdf >> "$LOGDIR/update.log" 2>&1
    "$UV" pip install --python "$PY" --reinstall-package pyside6-essentials --reinstall-package shiboken6 -r "$REQ" >> "$LOGDIR/update.log" 2>&1
    rm -rf "$RES/venv/lib/python3.12/site-packages/PySide6/Assistant.app" "$RES/venv/lib/python3.12/site-packages/PySide6/Designer.app" \
           "$RES/venv/lib/python3.12/site-packages/PySide6/Linguist.app" "$RES/venv/lib/python3.12/site-packages/PyObjCTest"
  fi
  find "$RES/venv" -name "__pycache__" -path "*/tests/*" -prune -exec rm -rf {} + 2>/dev/null
fi
exec "$RES/venv/bin/python" -X faulthandler main.py "$@" > "$LOGDIR/run.log" 2>&1
LAUNCH
chmod +x "$APP/Contents/MacOS/Forma"

# --- Info.plist
VERSION="$(grep __version__ "$SRC/forma/__init__.py" | cut -d'"' -f2)"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Forma</string>
  <key>NSHumanReadableCopyright</key><string>Created by Hatef Jafari Sharami</string>
  <key>CFBundleGetInfoString</key><string>Forma, created by Hatef Jafari Sharami</string>
  <key>CFBundleDisplayName</key><string>Forma</string>
  <key>CFBundleIdentifier</key><string>app.forma.cad</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>Forma</string>
  <key>CFBundleIconFile</key><string>Forma</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key><string>Forma Project</string>
      <key>CFBundleTypeExtensions</key><array><string>forma</string></array>
      <key>CFBundleTypeIconFile</key><string>Forma</string>
      <key>CFBundleTypeRole</key><string>Editor</string>
      <key>LSHandlerRank</key><string>Owner</string>
    </dict>
    <dict>
      <key>CFBundleTypeName</key><string>3D Model</string>
      <key>CFBundleTypeExtensions</key><array><string>3dm</string><string>step</string><string>stp</string><string>igs</string><string>iges</string></array>
      <key>CFBundleTypeRole</key><string>Editor</string>
      <key>LSHandlerRank</key><string>Alternate</string>
    </dict>
  </array>
</dict>
</plist>
PLIST
touch "$APP"
echo "Done: $APP"
echo "Drag Forma.app into Applications (or the Dock) and double-click it."
