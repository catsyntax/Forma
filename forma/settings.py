"""User settings and appearance themes (persisted to ~/.forma/settings.json)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".forma", "settings.json")
FONT = "Arial"

PRESETS = {
    "Dark": dict(mode="dark", window="#1e1f22", panel="#26272b", field="#1a1b1e", text="#e6e6e6", muted="#8c8f96",
                 border="#34363b", accent="#4c8dff", selection="#ffd23f",
                 view3d_top="#3a3d44", view3d_bottom="#202226", view2d="#1b1c1f", grid="#2c2e33", grid_major="#3b3e45",
                 preview="#3fd18b"),
    "Light": dict(mode="light", window="#f4f4f5", panel="#ffffff", field="#fafafa", text="#1b1b1d", muted="#6b6e75",
                  border="#dedfe3", accent="#2563eb", selection="#e8590c",
                  view3d_top="#ffffff", view3d_bottom="#d9dce2", view2d="#ffffff", grid="#eceef1", grid_major="#d3d6dc",
                  preview="#109a5a"),
    "Graphite": dict(mode="dark", window="#2b2d31", panel="#323439", field="#26282c", text="#ececec", muted="#9a9da4",
                     border="#40434a", accent="#7aa2f7", selection="#ffcc4d",
                     view3d_top="#5a5d64", view3d_bottom="#3a3c41", view2d="#2e3034", grid="#383a3f", grid_major="#474a51",
                     preview="#56d39b"),
}


@dataclass
class Settings:
    theme: str = "Dark"
    colors: dict = field(default_factory=dict)  # per-key overrides on top of the preset
    gradient_3d: bool = True
    back_lines_2d: str = "solid"     # Draft view: solid | dashed | hidden lines behind surfaces
    back_lines_3d: str = "solid"     # Model view: same
    ambient_occlusion: bool = False  # soft contact shadows in the 3D view (SSAO); can be slow on older GPUs
    font_size: int = 12
    grid_spacing: float = 100.0
    grid_extent: int = 50
    snap: bool = True
    ortho: bool = False
    osnap: bool = True
    show_grid: bool = True
    gumball: bool = True
    units: str = "mm"
    always_show_compute_dialog: bool = False
    compute_dialog_delay_ms: int = 400
    shortcuts: dict = field(default_factory=dict)   # action id -> key sequence ('' = none), overrides defaults

    def color(self, key):
        return self.colors.get(key) or PRESETS.get(self.theme, PRESETS["Dark"])[key]

    def palette(self):
        p = dict(PRESETS.get(self.theme, PRESETS["Dark"]))
        p.update({k: v for k, v in self.colors.items() if v})
        return p

    def save(self):
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @staticmethod
    def load():
        try:
            with open(SETTINGS_PATH) as f:
                d = json.load(f)
            s = Settings()
            for k, v in d.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            return s
        except Exception:
            return Settings()


def stylesheet(s: Settings) -> str:
    c = s.palette()
    fs = s.font_size
    return f"""
* {{ font-family: "{FONT}"; font-size: {fs}px; color: {c['text']}; }}
QMainWindow, QDialog {{ background: {c['window']}; }}
QWidget {{ background: {c['window']}; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {c['panel']}; padding: 6px 8px; font-weight: bold; border-bottom: 1px solid {c['border']}; }}
QFrame#Panel, QWidget#Panel {{ background: {c['panel']}; }}
QLabel#Heading {{ font-weight: bold; font-size: {fs + 1}px; background: transparent; }}
QLabel#Muted {{ color: {c['muted']}; background: transparent; }}
QLabel {{ background: transparent; }}
QToolBar {{ background: {c['panel']}; border: none; border-bottom: 1px solid {c['border']}; spacing: 2px; padding: 3px; }}
QToolBar#ViewBar {{ padding: 1px 4px; }}
QToolBar#ViewBar QToolButton {{ padding: 2px 8px; font-size: {max(9, s.font_size - 1)}px; }}
QToolButton, QPushButton {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 4px 9px; }}
QToolButton:hover, QPushButton:hover {{ background: {c['field']}; border-color: {c['border']}; }}
QToolButton:checked {{ background: {c['field']}; border-color: {c['accent']}; }}
QPushButton {{ border-color: {c['border']}; background: {c['panel']}; }}
QPushButton#Primary {{ background: {c['accent']}; color: white; border-color: {c['accent']}; font-weight: bold; }}
QPushButton#Persona {{ border: none; border-radius: 0; padding: 8px 16px; font-weight: normal; color: {c['muted']}; background: transparent; border-bottom: 2px solid transparent; }}
QPushButton#Persona:hover {{ color: {c['text']}; background: transparent; }}
QPushButton#Persona:checked {{ color: {c['text']}; font-weight: bold; border-bottom: 2px solid {c['accent']}; }}
QWidget#PersonaBar {{ background: {c['panel']}; border-bottom: 1px solid {c['border']}; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTreeWidget, QTableWidget {{
    background: {c['field']}; border: 1px solid {c['border']}; border-radius: 4px; padding: 3px;
    selection-background-color: {c['accent']}; selection-color: white; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {c['accent']}; }}
QHeaderView::section {{ background: {c['panel']}; border: none; border-bottom: 1px solid {c['border']}; padding: 4px; font-weight: bold; }}
QTabBar::tab {{ background: transparent; padding: 6px 12px; color: {c['muted']}; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {c['text']}; font-weight: bold; border-bottom: 2px solid {c['accent']}; }}
QTabWidget::pane {{ border: none; }}
QMenuBar {{ background: {c['panel']}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {c['accent']}; color: white; }}
QMenu {{ background: {c['panel']}; border: 1px solid {c['border']}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 12px; }}
QStatusBar {{ background: {c['panel']}; border-top: 1px solid {c['border']}; color: {c['muted']}; }}
QSplitter::handle {{ background: {c['border']}; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: transparent; width: 10px; height: 10px; }}
QScrollBar::handle {{ background: {c['border']}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QProgressBar {{ border: 1px solid {c['border']}; border-radius: 4px; background: {c['field']}; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}
QCheckBox {{ background: transparent; }}
QToolTip {{ background: {c['panel']}; color: {c['text']}; border: 1px solid {c['border']}; }}
"""
