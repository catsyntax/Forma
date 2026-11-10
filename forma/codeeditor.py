"""Script editor window: line numbers, Python highlighting, auto-indent, Run without closing, output pane.

Used by Python nodes (like Grasshopper's script editor) and by the ported tools, whose engine code
can be edited and run in place of the built-in version.
"""
from __future__ import annotations

import keyword
import re

from PySide6 import QtCore, QtGui, QtWidgets


# ------------------------------------------------------------------ highlighting
class PythonHighlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, doc, dark=True):
        super().__init__(doc)
        pal = {"kw": "#c678dd", "builtin": "#56b6c2", "str": "#98c379", "num": "#d19a66", "com": "#7f848e",
               "def": "#61afef", "deco": "#e5c07b", "self": "#e06c75"} if dark else \
              {"kw": "#7a1fa2", "builtin": "#00799a", "str": "#2e7d32", "num": "#b35c00", "com": "#8a8f98",
               "def": "#1565c0", "deco": "#9a6700", "self": "#c62828"}

        def fmt(c, bold=False, italic=False):
            f = QtGui.QTextCharFormat()
            f.setForeground(QtGui.QColor(c))
            if bold:
                f.setFontWeight(QtGui.QFont.Bold)
            f.setFontItalic(italic)
            return f
        builtins = ("print", "len", "range", "enumerate", "zip", "min", "max", "sum", "abs", "int", "float", "str",
                    "list", "dict", "set", "tuple", "sorted", "isinstance", "round", "map", "filter", "any", "all",
                    "open", "super", "type", "getattr", "setattr", "hasattr", "reversed")
        self.rules = [
            (re.compile(r"\b(" + "|".join(keyword.kwlist) + r")\b"), fmt(pal["kw"], True)),
            (re.compile(r"\b(" + "|".join(builtins) + r")\b"), fmt(pal["builtin"])),
            (re.compile(r"\b(self|cls)\b"), fmt(pal["self"])),
            (re.compile(r"\b[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?\b"), fmt(pal["num"])),
            (re.compile(r"(?<=\bdef\s)\w+|(?<=\bclass\s)\w+"), fmt(pal["def"], True)),
            (re.compile(r"^\s*@\w+"), fmt(pal["deco"])),
        ]
        self.str_fmt = fmt(pal["str"])
        self.com_fmt = fmt(pal["com"], italic=True)
        self.strings = re.compile(r"(?:[rbuRBU]{0,2})('([^'\\\n]|\\.)*'|\"([^\"\\\n]|\\.)*\")")
        self.tri = [re.compile("'''"), re.compile('"""')]

    def highlightBlock(self, text):
        for rx, f in self.rules:
            for m in rx.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), f)
        for m in self.strings.finditer(text):
            self.setFormat(m.start(), m.end() - m.start(), self.str_fmt)
        # comments (not inside strings)
        in_str, q = False, ""
        for i, ch in enumerate(text):
            if ch in "'\"" and (i == 0 or text[i - 1] != "\\"):
                if not in_str:
                    in_str, q = True, ch
                elif ch == q:
                    in_str = False
            elif ch == "#" and not in_str:
                self.setFormat(i, len(text) - i, self.com_fmt)
                break
        # triple-quoted strings across lines
        self.setCurrentBlockState(0)
        start = 0
        prev = self.previousBlockState()
        for k, rx in enumerate(self.tri, start=1):
            if prev == k:
                m = rx.search(text, 0)
                if m is None:
                    self.setFormat(0, len(text), self.str_fmt)
                    self.setCurrentBlockState(k)
                    return
                self.setFormat(0, m.end(), self.str_fmt)
                start = m.end()
        pos = start
        while True:
            hits = [(m.start(), k, m) for k, rx in enumerate(self.tri, start=1) if (m := rx.search(text, pos))]
            if not hits:
                break
            s0, k, m = min(hits, key=lambda h: h[0])
            end = self.tri[k - 1].search(text, m.end())
            if end is None:
                self.setFormat(s0, len(text) - s0, self.str_fmt)
                self.setCurrentBlockState(k)
                break
            self.setFormat(s0, end.end() - s0, self.str_fmt)
            pos = end.end()


# ------------------------------------------------------------------ editor widget
class _Gutter(QtWidgets.QWidget):
    def __init__(self, ed):
        super().__init__(ed)
        self.ed = ed

    def sizeHint(self):
        return QtCore.QSize(self.ed.gutter_width(), 0)

    def paintEvent(self, ev):
        self.ed.paint_gutter(ev)


class CodeEdit(QtWidgets.QPlainTextEdit):
    run_requested = QtCore.Signal()

    def __init__(self, text="", dark=True, parent=None):
        super().__init__(parent)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        font.setPointSize(max(11, font.pointSize()))
        self.setFont(font)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.setTabStopDistance(QtGui.QFontMetricsF(font).horizontalAdvance(" ") * 4)
        self.dark = dark
        self.gutter = _Gutter(self)
        self.blockCountChanged.connect(lambda _: self.setViewportMargins(self.gutter_width(), 0, 0, 0))
        self.updateRequest.connect(self._update_gutter)
        self.cursorPositionChanged.connect(self._highlight_line)
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)
        self.highlighter = PythonHighlighter(self.document(), dark)
        self.setPlainText(text)
        self._highlight_line()

    def gutter_width(self):
        digits = len(str(max(1, self.blockCount())))
        return 14 + self.fontMetrics().horizontalAdvance("9") * max(3, digits)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        cr = self.contentsRect()
        self.gutter.setGeometry(QtCore.QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def _update_gutter(self, rect, dy):
        if dy:
            self.gutter.scroll(0, dy)
        else:
            self.gutter.update(0, rect.y(), self.gutter.width(), rect.height())

    def paint_gutter(self, ev):
        p = QtGui.QPainter(self.gutter)
        p.fillRect(ev.rect(), QtGui.QColor("#21252b" if self.dark else "#f0f1f3"))
        block = self.firstVisibleBlock()
        n = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        cur = self.textCursor().blockNumber()
        while block.isValid() and top <= ev.rect().bottom():
            if block.isVisible() and bottom >= ev.rect().top():
                p.setPen(QtGui.QColor("#d7dae0" if n == cur else "#5c6370") if self.dark else
                         QtGui.QColor("#202124" if n == cur else "#9aa0a6"))
                p.drawText(0, top, self.gutter.width() - 6, self.fontMetrics().height(), QtCore.Qt.AlignRight, str(n + 1))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            n += 1
        p.end()

    def _highlight_line(self):
        sel = QtWidgets.QTextEdit.ExtraSelection()
        sel.format.setBackground(QtGui.QColor("#2c313a" if self.dark else "#eef3fb"))
        sel.format.setProperty(QtGui.QTextFormat.FullWidthSelection, True)
        sel.cursor = self.textCursor()
        sel.cursor.clearSelection()
        self.setExtraSelections([sel] + getattr(self, "_error_sel", []))

    def mark_error_line(self, line):
        self._error_sel = []
        if line:
            block = self.document().findBlockByNumber(line - 1)
            if block.isValid():
                sel = QtWidgets.QTextEdit.ExtraSelection()
                sel.format.setBackground(QtGui.QColor("#5a2a2a" if self.dark else "#fde2e2"))
                sel.format.setProperty(QtGui.QTextFormat.FullWidthSelection, True)
                sel.cursor = QtGui.QTextCursor(block)
                self._error_sel = [sel]
        self._highlight_line()

    def keyPressEvent(self, ev):
        k, mod = ev.key(), ev.modifiers()
        if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and mod & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier):
            self.run_requested.emit()
            return
        if k == QtCore.Qt.Key_Tab and not self.textCursor().hasSelection():
            self.insertPlainText("    ")
            return
        if k in (QtCore.Qt.Key_Tab, QtCore.Qt.Key_Backtab) and self.textCursor().hasSelection():
            self._indent(k == QtCore.Qt.Key_Tab)
            return
        if k == QtCore.Qt.Key_Backtab:
            self._indent(False)
            return
        if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            line = self.textCursor().block().text()
            indent = re.match(r"\s*", line).group(0)
            if line.rstrip().endswith(":"):
                indent += "    "
            super().keyPressEvent(ev)
            self.insertPlainText(indent)
            return
        if k == QtCore.Qt.Key_Backspace and not self.textCursor().hasSelection():
            c = self.textCursor()
            before = c.block().text()[:c.positionInBlock()]
            if before and not before.strip() and len(before) % 4 == 0:
                for _ in range(4):
                    c.deletePreviousChar()
                return
        if k == QtCore.Qt.Key_Slash and mod & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier):
            self._toggle_comment()
            return
        super().keyPressEvent(ev)

    def _lines(self):
        c = self.textCursor()
        doc = self.document()
        a = doc.findBlock(c.selectionStart()).blockNumber()
        b = doc.findBlock(c.selectionEnd()).blockNumber()
        return a, b

    def _indent(self, inc):
        a, b = self._lines()
        c = self.textCursor()
        c.beginEditBlock()
        for n in range(a, b + 1):
            blk = self.document().findBlockByNumber(n)
            cur = QtGui.QTextCursor(blk)
            if inc:
                cur.insertText("    ")
            else:
                t = blk.text()
                k = len(t) - len(t.lstrip(" "))
                for _ in range(min(4, k)):
                    cur.deleteChar()
        c.endEditBlock()

    def _toggle_comment(self):
        a, b = self._lines()
        blocks = [self.document().findBlockByNumber(n) for n in range(a, b + 1)]
        uncomment = all(bl.text().lstrip().startswith("#") or not bl.text().strip() for bl in blocks)
        c = self.textCursor()
        c.beginEditBlock()
        for bl in blocks:
            t = bl.text()
            if not t.strip():
                continue
            cur = QtGui.QTextCursor(bl)
            k = len(t) - len(t.lstrip())
            cur.setPosition(bl.position() + k)
            if uncomment:
                n = 2 if t[k:].startswith("# ") else 1
                for _ in range(n):
                    cur.deleteChar()
            else:
                cur.insertText("# ")
        c.endEditBlock()


# ------------------------------------------------------------------ window
class ScriptWindow(QtWidgets.QDialog):
    """Non-modal editor. `apply(code, inputs, outputs, list_inputs)` stores the script and starts a solve;
    `fetch()` returns (output text, error text or None) once the result is in, else None while running."""

    def __init__(self, parent, title, code, apply, fetch, ports=None, reset_code=None, note="", dark=True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.resize(980, 760)
        self._apply, self._fetch, self._reset = apply, fetch, reset_code
        self._saved = code
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        run = QtWidgets.QPushButton("▶  Run")
        run.setObjectName("Primary")
        run.setToolTip("Apply the code and compute (Ctrl+Enter). The window stays open.")
        run.clicked.connect(self.run)
        ok = QtWidgets.QPushButton("Save && close")
        ok.clicked.connect(self.save_close)
        rev = QtWidgets.QPushButton("Revert")
        rev.setToolTip("Go back to the last saved code")
        rev.clicked.connect(lambda: self.ed.setPlainText(self._saved))
        bar.addWidget(run)
        bar.addWidget(ok)
        bar.addWidget(rev)
        if reset_code is not None:
            rs = QtWidgets.QPushButton("Reset to built-in code")
            rs.clicked.connect(self._reset_builtin)
            bar.addWidget(rs)
        bar.addStretch(1)
        self.find = QtWidgets.QLineEdit()
        self.find.setPlaceholderText("Find (Enter)")
        self.find.setMaximumWidth(220)
        self.find.returnPressed.connect(self._find)
        bar.addWidget(self.find)
        lay.addLayout(bar)
        self.ins = self.outs = self.lists = None
        if ports is not None:
            form = QtWidgets.QFormLayout()
            self.ins = QtWidgets.QLineEdit(", ".join(ports.get("inputs", [])))
            self.outs = QtWidgets.QLineEdit(", ".join(ports.get("outputs", [])))
            self.lists = QtWidgets.QLineEdit(", ".join(ports.get("list_inputs", [])))
            self.lists.setPlaceholderText("inputs that receive whole lists (others run once per item)")
            form.addRow("Inputs", self.ins)
            form.addRow("Outputs", self.outs)
            form.addRow("List inputs", self.lists)
            lay.addLayout(form)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.ed = CodeEdit(code, dark)
        self.ed.run_requested.connect(self.run)
        split.addWidget(self.ed)
        outw = QtWidgets.QWidget()
        ol = QtWidgets.QVBoxLayout(outw)
        ol.setContentsMargins(0, 4, 0, 0)
        self.status = QtWidgets.QLabel(note or "Ctrl+Enter runs. Ctrl+/ comments lines. Tab / Shift+Tab indents.")
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        ol.addWidget(self.status)
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setFont(self.ed.font())
        self.out.setPlaceholderText("print() output and errors appear here after Run.")
        ol.addWidget(self.out)
        split.addWidget(outw)
        split.setSizes([560, 160])
        lay.addWidget(split, 1)
        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(200)
        self._poll.timeout.connect(self._check)
        QtGui.QShortcut(QtGui.QKeySequence.Find, self, activated=lambda: self.find.setFocus())
        QtGui.QShortcut(QtGui.QKeySequence.Save, self, activated=self.run)

    def _ports(self):
        if self.ins is None:
            return None
        split = lambda s: [x.strip() for x in s.split(",") if x.strip().isidentifier()]
        return split(self.ins.text()), split(self.outs.text()), split(self.lists.text())

    def run(self):
        code = self.ed.toPlainText()
        try:
            compile(code, "<script>", "exec")
        except SyntaxError as e:
            self.out.setPlainText(f"SyntaxError: {e.msg} (line {e.lineno})")
            self.ed.mark_error_line(e.lineno)
            self.status.setText("Not run: fix the syntax error first.")
            return False
        self.ed.mark_error_line(None)
        self._apply(code, self._ports())
        self._saved = code
        self.status.setText("Running…")
        self.out.setPlainText("")
        self._ticks = 0
        self._poll.start()
        return True

    def _check(self):
        self._ticks += 1
        r = self._fetch()
        if r is None:
            if self._ticks % 5 == 0:
                self.status.setText("Running… " + "." * (self._ticks // 5 % 4))
            return
        self._poll.stop()
        text, err = r
        if err:
            self.status.setText("Error")
            self.out.setPlainText(err)
            m = re.findall(r'line (\d+)', err)
            self.ed.mark_error_line(int(m[-1]) if m else None)
        else:
            self.status.setText("Done.")
            self.out.setPlainText(text or "(no printed output)")

    def save_close(self):
        if self.ed.toPlainText() != self._saved:
            if not self.run():
                return
        self.close()

    def _reset_builtin(self):
        if QtWidgets.QMessageBox.question(self, "Reset code", "Replace the code with the built-in version?") \
                == QtWidgets.QMessageBox.Yes:
            self.ed.setPlainText(self._reset())

    def _find(self):
        t = self.find.text()
        if t and not self.ed.find(t):
            self.ed.moveCursor(QtGui.QTextCursor.Start)
            self.ed.find(t)
