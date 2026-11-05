"""Start Forma:  python -m forma [file.forma | file.3dm | file.step | ...]   (or the `forma` command)."""
import faulthandler
import multiprocessing
import os
import signal
import sys
import time


def smoke_test():
    """Headless check used by the build pipeline: libraries load, the NURBS kernel and nodes work."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from forma import brep as B, parametric as PA, __version__
    import forma.app  # noqa: F401  (Qt, VTK and every workspace import)
    from forma.nodes.registry import NODE_TYPES
    d = B.boolean([B.box((0, 0, 0), 1000, 1000, 1000)], [B.box((250, 250, -10), 500, 500, 1100)], "difference")
    assert abs(d.volume() - 7.5e8) < 1e3
    assert len(PA.stadium()) > 3
    r = NODE_TYPES["U-Value"].fn(Thicknesses=[100], Conductivities=[0.035])
    assert r["U"] > 0
    # the node solver runs in a separate process: check that works in this build too
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    from forma.selftest import worker_check
    p = ctx.Process(target=worker_check, args=(q,))
    p.start()
    vol = q.get(timeout=120)
    p.join(30)
    assert abs(vol - 1e9) < 1e3, vol
    print(f"Forma {__version__} smoke test OK", flush=True)
    return 0


def run():
    multiprocessing.freeze_support()
    if "--smoke-test" in sys.argv:
        return smoke_test()
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
