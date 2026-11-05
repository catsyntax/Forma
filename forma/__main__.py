"""Start Forma:  python -m forma [file.forma | file.3dm | file.step | ...]   (or the `forma` command)."""
import faulthandler
import multiprocessing
import os
import signal
import sys
import time


def run():
    multiprocessing.freeze_support()
    t0 = time.time()
    faulthandler.enable(all_threads=True)
    for sig in ("SIGTRAP", "SIGBUS", "SIGILL"):
        if hasattr(signal, sig):
            try:
                faulthandler.register(getattr(signal, sig), all_threads=True, chain=True)
            except Exception:
                pass
    os.environ.setdefault("FORMA_TRACE", "1")
    print(f"[forma] Python {sys.version.split()[0]} on {sys.platform} - starting...", flush=True)
    faulthandler.dump_traceback_later(300, exit=False)

    def _on_term(signum, frame):
        print(f"[forma] received signal {signum} (parent pid {os.getppid()}) - quitting", flush=True)
        os._exit(0)
    for _s in ("SIGTERM", "SIGHUP", "SIGINT"):
        if hasattr(signal, _s):
            try:
                signal.signal(getattr(signal, _s), _on_term)
            except Exception:
                pass
    import atexit
    atexit.register(lambda: print("[forma] process exiting", flush=True))

    # Show a splash immediately, before the heavy 3D / NURBS libraries load.
    from PySide6 import QtCore, QtGui, QtWidgets
    from forma.bootstrap import create_qapp, _FileOpenFilter  # light: the heavy libraries load after the splash
    app = create_qapp(sys.argv)
    files = _FileOpenFilter()
    app.installEventFilter(files)
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "forma_icon.png")
    pm = QtGui.QPixmap(460, 220)
    pm.fill(QtGui.QColor("#1e1f22"))
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    if os.path.exists(icon_path):
        p.drawPixmap(28, 46, QtGui.QPixmap(icon_path).scaled(128, 128, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        app.setWindowIcon(QtGui.QIcon(icon_path))
    p.setPen(QtGui.QColor("#e6e6e6"))
    p.setFont(QtGui.QFont("Arial", 30, QtGui.QFont.Bold))
    p.drawText(176, 112, "Forma")
    p.setPen(QtGui.QColor("#8c8f96"))
    p.setFont(QtGui.QFont("Arial", 12))
    p.drawText(178, 140, "Draft · Model · Components · Nodes")
    p.end()
    splash = QtWidgets.QSplashScreen(pm)
    splash.show()
    splash.showMessage("Loading 3D and NURBS libraries...", QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft, QtGui.QColor("#8c8f96"))
    app.processEvents()

    from forma.app import main
    print(f"[forma] libraries loaded in {time.time() - t0:.1f}s - opening window...", flush=True)
    splash.showMessage("Opening workspace...", QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft, QtGui.QColor("#8c8f96"))
    app.processEvents()
    return main(sys.argv, splash=splash, file_filter=files)


if __name__ == "__main__":
    sys.exit(run())
