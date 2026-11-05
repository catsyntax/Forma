#!/bin/bash
# Adds Forma to your applications menu (run once from inside the unpacked Forma folder).
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/forma.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Forma
Comment=NURBS CAD: drafting, 3D modeling, components and nodes
Exec="$HERE/Forma" %f
Icon=$HERE/forma_icon.png
Terminal=false
Categories=Graphics;3DGraphics;Engineering;
DESK
echo "Forma added to your applications menu. You can also start it with: $HERE/Forma"
