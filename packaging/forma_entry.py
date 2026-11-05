"""Entry point for the packaged (PyInstaller) app."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()     # the node solver runs in a worker process
    from forma.__main__ import run
    sys.exit(run())
