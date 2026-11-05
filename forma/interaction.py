"""Interaction controller (runs commands, resolves snaps) and the command line widget."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import commands as K
from .geometry import P


class Interaction(QtCore.QObject):
    changed = QtCore.Signal()          # prompt / state changed
    preview_changed = QtCore.Signal()
    message = QtCore.Signal(str)

    def __init__(self, doc, settings, app=None):
        super().__init__()
        self.doc = doc
        self.settings = settings
        self.app = app
        self.runner: K.Runner | None = None
        self.space = "3d"
        self.last_command = None
        self.cursor = None          # last resolved cursor point (world)
        self.snap_label = ""
        self.last_point = None

    # ------------------------------------------------------------ state
    @property
    def request(self):
        return self.runner.request if self.runner and not self.runner.done else None

    @property
    def active(self):
        return self.request is not None

    def prompt(self):
        r = self.request
        if r is None:
            return "Command"
        txt = r.prompt
        if isinstance(r, K.GetNumber) and r.default is not None:
            txt += f" <{r.default:g}>"
        if isinstance(r, K.GetChoice):
            txt += " (" + " / ".join(r.choices) + ")" + (f" <{r.default}>" if r.default else "")
        if isinstance(r, K.GetString) and r.default:
            txt += f" <{r.default}>"
        if isinstance(r, K.GetObjects):
            n = len(self.doc.selected(self.space))
            txt += f"  [{n} selected, Enter to accept]"
        return txt

    # ------------------------------------------------------------ run
    def start(self, name, space=None, **kw):
        cmd = K.find_command(name)
        if cmd is None:
            self.message.emit(f"Unknown command: {name}")
            return False
        space = space or self.space
        if space not in cmd.spaces:
            self.message.emit(f"{cmd.name} is not available in the {'Draft' if space == '2d' else 'Model'} workspace.")
            return False
        if self.runner and not self.runner.done:
            self.runner.cancel()
        self.space = space
        ctx = K.Context(self.doc, space, self.app)
        if space == "2d":
            ctx.view_R = getattr(self, "view_R", None)
        ctx.last_point = self.last_point
        self.runner = K.Runner(ctx, cmd, **kw)
        self.message.emit(cmd.name)
        if cmd.name not in ("Undo", "Redo"):
            self.last_command = cmd.name
        self.runner.start()
        self._after()
        return True

    def feed(self, value):
        if not self.runner or self.runner.done:
            return
        if isinstance(value, np.ndarray):
            self.last_point = value
            self.runner.ctx.last_point = value
        self.runner.feed(value)
        self._after()

    def cancel(self):
        if self.runner and not self.runner.done:
            self.runner.cancel()
            self.message.emit("Command cancelled")
        self._after()

    def _after(self):
        self.changed.emit()
        self.preview_changed.emit()

    # ------------------------------------------------------------ inputs
    def submit_empty(self):
        r = self.request
        if r is None:
            if self.last_command:
                self.start(self.last_command)
            return
        if isinstance(r, K.GetPoint):
            if r.allow_none:
                self.feed(None)
        elif isinstance(r, K.GetObjects):
            sel = [g for g in self.doc.selected(self.space) if r.filter is None or r.filter(g)]
            if r.single:
                if len(sel) >= 1:
                    self.feed(sel[0])
            elif sel:
                self.feed(sel)
        elif isinstance(r, K.GetNumber):
            if r.default is not None:
                self.feed(float(r.default))
        elif isinstance(r, K.GetChoice):
            if r.default is not None:
                self.feed(r.default)
        elif isinstance(r, K.GetString):
            self.feed(r.default)

    def submit_text(self, text):
        text = text.strip()
        if not text:
            return self.submit_empty()
        r = self.request
        try:
            if r is None:
                self.start(text)
            elif isinstance(r, K.GetPoint):
                if text.upper() in [o.upper() for o in r.options]:
                    self.feed(text.upper())
                    return
                base = r.base
                if base is not None and "," not in text and "<" not in text:
                    # distance constraint along the cursor direction
                    d = K.eval_number(text)
                    direction = (self.cursor - base) if self.cursor is not None else np.array([1.0, 0, 0])
                    if np.linalg.norm(direction) < 1e-12:
                        direction = np.array([1.0, 0, 0])
                    self.feed(base + direction / np.linalg.norm(direction) * d)
                    return
                self.feed(K.parse_point(text, base, self.last_point, self.space))
            elif isinstance(r, K.GetNumber):
                v = K.eval_number(text)
                self.feed(int(round(v)) if r.integer else v)
            elif isinstance(r, K.GetChoice):
                t = text.lower()
                m = [c for c in r.choices if c.lower() == t] or [c for c in r.choices if c.lower().startswith(t)]
                if m:
                    self.feed(m[0])
                else:
                    self.message.emit("Choose one of: " + ", ".join(r.choices))
            elif isinstance(r, K.GetString):
                self.feed(text)
            elif isinstance(r, K.GetObjects):
                if text.lower() == "all":
                    self.doc.select([g.id for g in self.doc.selectable(self.space) if r.filter is None or r.filter(g)])
                    self.submit_empty()
        except Exception as e:
            self.message.emit(f"Invalid input: {e}")

    def click_point(self, p):
        """A resolved point was clicked in a view."""
        r = self.request
        if isinstance(r, K.GetPoint):
            self.feed(P(p))
        elif isinstance(r, K.GetNumber) and r.base is not None:
            v = float(np.linalg.norm(P(p) - r.base))
            self.feed(int(round(v)) if r.integer else v)

    def wants_window(self):
        r = self.request
        return isinstance(r, K.GetPoint) and getattr(r, "window", False)

    def window_pick(self, inside, crossing):
        """A selection window was dragged while a command accepts one (e.g. Trim)."""
        if self.wants_window():
            self.feed(("window", inside, bool(crossing)))

    def wants_point(self):
        r = self.request
        return isinstance(r, K.GetPoint) or (isinstance(r, K.GetNumber) and r.base is not None)

    def wants_objects(self):
        return isinstance(self.request, K.GetObjects)

    def selection_filter(self):
        r = self.request
        return r.filter if isinstance(r, K.GetObjects) else None

    def object_clicked_in_select_mode(self):
        """After a click-select during GetObjects(single), accept immediately."""
        r = self.request
        if isinstance(r, K.GetObjects) and r.single:
            sel = [g for g in self.doc.selected(self.space) if r.filter is None or r.filter(g)]
            if len(sel) == 1:
                self.feed(sel[0])

    # ------------------------------------------------------------ snapping
    def constrain(self, p, base_override=None):
        """Apply ortho relative to request base."""
        r = self.request
        base = base_override if base_override is not None else (getattr(r, "base", None) if r else None)
        if self.settings.ortho and base is not None:
            d = p - base
            if self.space == "2d":
                n = getattr(self, "plane_normal", np.array([0.0, 0, 1]))
                k = int(np.argmax(np.abs(d - n * np.dot(d, n))))
            else:
                k = int(np.argmax(np.abs(d)))
            q = base.copy()
            q[k] = p[k]
            return q
        return p

    def grid_snap(self, p):
        if not self.settings.snap:
            return p
        s = self.settings.grid_spacing / 2 if self.space == "2d" else self.settings.grid_spacing
        q = p.copy()
        if self.space == "2d":
            # snap the two in-plane axes of the current Draft view
            n = getattr(self, "plane_normal", np.array([0.0, 0, 1]))
            for k in range(3):
                if abs(n[k]) < 0.5:
                    q[k] = round(p[k] / s) * s
        else:
            q[:2] = np.round(p[:2] / s) * s
        return q

    def preview(self):
        r = self.request
        if r is None or self.cursor is None:
            return []
        try:
            if isinstance(r, K.GetPoint) and r.preview:
                return r.preview(self.cursor) or []
            if isinstance(r, K.GetNumber) and r.preview and r.base is not None:
                return r.preview(float(np.linalg.norm(self.cursor - r.base))) or []
        except Exception:
            return []
        return []


class CommandLine(QtWidgets.QWidget):
    """History + prompt + input with autocomplete. Space and Enter both submit."""

    def __init__(self, interaction: Interaction, parent=None):
        super().__init__(parent)
        self.ix = interaction
        self.setObjectName("Panel")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 6)
        lay.setSpacing(2)
        self.history = QtWidgets.QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setMaximumBlockCount(500)
        self.history.setFixedHeight(54)
        self.history.setFrameShape(QtWidgets.QFrame.NoFrame)
        row = QtWidgets.QHBoxLayout()
        self.prompt = QtWidgets.QLabel("Command:")
        self.prompt.setObjectName("Heading")
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText("Type a command (e.g. Line, Box, Extrude) or Ctrl+K to search")
        row.addWidget(self.prompt)
        row.addWidget(self.edit, 1)
        lay.addWidget(self.history)
        lay.addLayout(row)
        self.cmd_history = []
        self.hist_idx = 0
        self.model = QtCore.QStringListModel()
        self.completer = QtWidgets.QCompleter(self.model, self)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.completer.setFilterMode(QtCore.Qt.MatchContains)
        self.completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.completer.activated.connect(self._completed)
        self.edit.setCompleter(self.completer)
        self.edit.installEventFilter(self)
        self.edit.textEdited.connect(self._edited)
        self.ix.changed.connect(self.refresh)
        self.ix.message.connect(self.log)
        self.refresh()

    def _edited(self, t):
        if self.ix.active:
            self.completer.popup().hide()
            return
        self.model.setStringList(K.complete(t, self.ix.space)[:40] if t else [])

    def _completed(self, text):
        QtCore.QTimer.singleShot(0, lambda: self._submit(text))

    def log(self, text):
        self.history.appendPlainText(text)
        self.history.verticalScrollBar().setValue(self.history.verticalScrollBar().maximum())

    def refresh(self):
        self.prompt.setText(self.ix.prompt() + ":")

    def _submit(self, text=None):
        t = self.edit.text() if text is None else text
        self.edit.clear()
        self.completer.popup().hide()
        if t.strip():
            self.cmd_history.append(t)
            self.hist_idx = len(self.cmd_history)
        if t.strip() and not self.ix.active:
            pass
        elif t.strip():
            self.log(f"  {self.ix.request.prompt}: {t}")
        self.ix.submit_text(t)

    def eventFilter(self, obj, ev):
        if obj is self.edit and ev.type() == QtCore.QEvent.KeyPress:
            k = ev.key()
            if self.completer.popup().isVisible() and k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                return False
            if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self._submit()
                return True
            if k == QtCore.Qt.Key_Space and (not self.edit.text().strip() or not self.ix.active):
                self._submit()
                return True
            if k == QtCore.Qt.Key_Escape:
                self.edit.clear()
                self.ix.cancel()
                return True
            if k == QtCore.Qt.Key_Up and self.cmd_history and not self.completer.popup().isVisible():
                self.hist_idx = max(0, self.hist_idx - 1)
                self.edit.setText(self.cmd_history[self.hist_idx])
                return True
            if k == QtCore.Qt.Key_Down and self.cmd_history and not self.completer.popup().isVisible():
                self.hist_idx = min(len(self.cmd_history), self.hist_idx + 1)
                self.edit.setText(self.cmd_history[self.hist_idx] if self.hist_idx < len(self.cmd_history) else "")
                return True
        return super().eventFilter(obj, ev)

    def forward_key(self, ev: QtGui.QKeyEvent):
        """Views forward typed characters here so you can type commands anywhere."""
        k = ev.key()
        if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) or (k == QtCore.Qt.Key_Space and not self.edit.text()):
            self._submit()
            return True
        if k == QtCore.Qt.Key_Escape:
            self.edit.clear()
            self.ix.cancel()
            return True
        if ev.text() and ev.text().isprintable() and not (ev.modifiers() & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier)):
            self.edit.setFocus()
            self.edit.setText(self.edit.text() + ev.text())
            self._edited(self.edit.text())
            return True
        if k == QtCore.Qt.Key_Backspace and self.edit.text():
            self.edit.setText(self.edit.text()[:-1])
            return True
        return False
