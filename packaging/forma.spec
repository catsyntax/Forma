# PyInstaller spec for the ready-to-install Forma apps (built by .github/workflows/release.yml).
#   pyinstaller packaging/forma.spec --noconfirm
# Produces dist/Forma/ (Windows, Linux) or dist/Forma.app (macOS).
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
datas, binaries, hidden = [], [], []
for pkg in ("OCP", "cadquery_ocp", "rhino3dm", "pyvista", "pyvistaqt", "trimesh", "ezdxf", "svgelements",
            "collada", "networkx"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hidden += h
    except Exception:
        pass
hidden += collect_submodules("vtkmodules") + collect_submodules("forma") + ["qtpy", "scipy.spatial", "lxml.etree"]
datas += collect_data_files("vtkmodules")
datas += [(os.path.join(ROOT, "forma", "resources"), "forma/resources"),
          (os.path.join(ROOT, "examples"), "examples")]

a = Analysis([os.path.join(ROOT, "packaging", "forma_entry.py")],
             pathex=[ROOT], binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "matplotlib.tests", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
                       "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml", "PyQt5", "PyQt6", "IPython",
                       "cv2", "imageio_ffmpeg", "imageio"],
             noarchive=False)
pyz = PYZ(a.pure)
icon = os.path.join(ROOT, "packaging", "Forma.icns" if sys.platform == "darwin" else "forma_icon.ico")
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Forma", console=False, icon=icon,
          upx=False, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Forma", upx=False)
if sys.platform == "darwin":
    app = BUNDLE(coll, name="Forma.app", icon=icon, bundle_identifier="app.forma.cad",
                 info_plist={"CFBundleName": "Forma", "CFBundleDisplayName": "Forma",
                             "NSHumanReadableCopyright": "Created by catsyntax",
                             "NSHighResolutionCapable": True,
                             "CFBundleShortVersionString": "0.6.0",
                             "CFBundleDocumentTypes": [{"CFBundleTypeName": "Forma project",
                                                        "CFBundleTypeExtensions": ["forma"],
                                                        "CFBundleTypeRole": "Editor"}]})
