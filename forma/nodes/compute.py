"""Runs node solves in a separate process and shows a cancellable progress dialog."""
from __future__ import annotations

import multiprocessing as mp
import queue
import time

from PySide6 import QtCore, QtWidgets

from .solver import worker_main


class ComputeManager(QtCore.QObject):
    node_started = QtCore.Signal(str)
    node_progress = QtCore.Signal(str, float)
    node_result = QtCore.Signal(str, object, float)
    node_error = QtCore.Signal(str, str)
    finished = QtCore.Signal(bool)  # cancelled?
    disable_node = QtCore.Signal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.inq = self.outq = None
        self.job = 0
        self.busy = False
        self.pending = None
        self.current_nodes = []
        self.running_node = None
        self.started_at = 0.0
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._poll)
        self.dialog = None
        self.names = {}

    def _ensure_worker(self):
        if self.proc is None or not self.proc.is_alive():
            self.inq, self.outq = self.ctx.Queue(), self.ctx.Queue()
            self.proc = self.ctx.Process(target=worker_main, args=(self.inq, self.outq), daemon=True)
            self.proc.start()

    def solve(self, graph, cache, dirty, refs, defs, names=None):
        if self.busy:
            self.pending = (graph, cache, dirty, refs, defs, names)
            return
        self._ensure_worker()
        self.job += 1
        self.busy = True
        self.names = names or {}
        self.current_nodes = list(dirty)
        self.started_at = time.time()
        self.running_node = None
        self.inq.put((self.job, graph, cache, list(dirty), refs, defs))
        self.timer.start()
        delay = 0 if self.settings.always_show_compute_dialog else self.settings.compute_dialog_delay_ms
        QtCore.QTimer.singleShot(delay, lambda j=self.job: self._maybe_dialog(j))

    def _maybe_dialog(self, job):
        if self.busy and job == self.job:
            if self.dialog is None:
                self.dialog = ComputeDialog(self, self.parent() if isinstance(self.parent(), QtWidgets.QWidget) else None)
            self.dialog.begin(self.current_nodes, self.names)
            self.dialog.show()
            self.dialog.raise_()

    def _poll(self):
        if self.outq is None:
            return
        for _ in range(200):
            try:
                msg = self.outq.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            job, kind = msg[0], msg[1]
            if job != self.job:
                continue
            if kind == "start":
                self.running_node = msg[2]
                self.node_started.emit(msg[2])
                if self.dialog:
                    self.dialog.node_state(msg[2], "Running")
            elif kind == "progress":
                self.node_progress.emit(msg[2], msg[3])
                if self.dialog:
                    self.dialog.progress(msg[2], msg[3])
            elif kind == "result":
                self.node_result.emit(msg[2], msg[3], msg[4])
                if self.dialog:
                    self.dialog.node_state(msg[2], f"Done  {msg[4]:.2f}s")
            elif kind == "error":
                self.node_error.emit(msg[2], msg[3])
                if self.dialog:
                    self.dialog.node_state(msg[2], "Error")
            elif kind == "fatal":
                self.node_error.emit("", msg[2])
            elif kind == "done":
                self._finish(False)
                return
        if self.dialog and self.dialog.isVisible():
            self.dialog.tick(time.time() - self.started_at, self.running_node)
        if self.busy and self.proc is not None and not self.proc.is_alive():
            self._finish(True)

    def _finish(self, cancelled):
        self.busy = False
        self.timer.stop()
        if self.dialog:
            self.dialog.hide()
        self.finished.emit(cancelled)
        if self.pending is not None:
            p, self.pending = self.pending, None
            self.solve(*p)

    def cancel(self):
        """Kill the worker. The next solve starts a fresh one."""
        if not self.busy:
            return None
        node = self.running_node
        self.pending = None
        if self.proc is not None:
            self.proc.terminate()
            self.proc.join(1.0)
        self.proc = None
        self._finish(True)
        return node

    def shutdown(self):
        try:
            if self.proc is not None and self.proc.is_alive():
                self.inq.put(None)
                self.proc.join(0.5)
                if self.proc.is_alive():
                    self.proc.terminate()
        except Exception:
            pass


class ComputeDialog(QtWidgets.QDialog):
    """Non-modal popup that lists node calculations with progress and a Cancel button."""
    disable_requested = QtCore.Signal(str)

    def __init__(self, mgr: ComputeManager, parent=None):
        super().__init__(parent, QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint)
        self.mgr = mgr
        self.setWindowTitle("Calculating")
        self.resize(420, 300)
        lay = QtWidgets.QVBoxLayout(self)
        self.title = QtWidgets.QLabel("Calculating...")
        self.title.setObjectName("Heading")
        lay.addWidget(self.title)
        self.current = QtWidgets.QLabel("")
        self.current.setObjectName("Muted")
        lay.addWidget(self.current)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        lay.addWidget(self.bar)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Node", "Status"])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.table)
        hint = QtWidgets.QLabel("If a calculation stops responding, cancel it. The rest of the app keeps working.")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        row = QtWidgets.QHBoxLayout()
        row.addStretch()
        self.disable_btn = QtWidgets.QPushButton("Cancel and disable node")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setObjectName("Primary")
        row.addWidget(self.disable_btn)
        row.addWidget(self.cancel_btn)
        lay.addLayout(row)
        self.cancel_btn.clicked.connect(self._cancel)
        self.disable_btn.clicked.connect(self._cancel_disable)
        self.rows = {}

    def begin(self, nids, names):
        self.rows = {}
        self.table.setRowCount(len(nids))
        for r, nid in enumerate(nids):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(names.get(nid, nid)))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem("Waiting"))
            self.rows[nid] = r
        self.names = names
        self.bar.setValue(0)

    def node_state(self, nid, state):
        r = self.rows.get(nid)
        if r is not None:
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(state))
        if state == "Running":
            self.bar.setValue(0)

    def progress(self, nid, f):
        self.bar.setValue(int(f * 1000))

    def tick(self, elapsed, nid):
        nm = self.names.get(nid, nid or "")
        self.title.setText(f"Calculating  ({elapsed:.1f}s)")
        self.current.setText(f"Current: {nm}" if nid else "Starting worker...")
        if elapsed > 10:
            self.title.setText(f"Still calculating  ({elapsed:.0f}s). Not responding? Press Cancel.")

    def _cancel(self):
        self.mgr.cancel()

    def _cancel_disable(self):
        nid = self.mgr.cancel()
        if nid:
            self.mgr.disable_node.emit(nid)

    def closeEvent(self, ev):
        self.mgr.cancel()
        super().closeEvent(ev)
