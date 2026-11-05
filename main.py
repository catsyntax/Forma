"""Run Forma from a source checkout:  python main.py [file]   (same as: python -m forma [file])"""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from forma.__main__ import run
    sys.exit(run())
