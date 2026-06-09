"""
mapgen_app.py — Premium macOS desktop app for mapgen(ai)
=========================================================
PySide6 native app with embedded interactive map viewer.
Dark theme, PIF red accent, drag-and-drop file loading.

Run:
    .venv/bin/python mapgen_app.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from PySide6.QtCore import (
    Qt, QThread, Signal, QPropertyAnimation, QEasingCurve,
    QSize, QTimer, QUrl, QPoint,
)
from PySide6.QtGui import (
    QColor, QDragEnterEvent, QDropEvent, QFont, QFontDatabase,
    QLinearGradient, QPainter, QPalette, QPixmap, QIcon,
    QBrush, QPen,
)
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton, QSizePolicy,
    QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
    QComboBox, QFrame, QSplitter, QScrollArea, QGraphicsOpacityEffect,
    QGraphicsDropShadowEffect,
)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from data_analyzer import clean_dataframe, profile_data, suggest_emoji, pivot_long_to_wide
from state_agent   import run as agent2_run
from map_engine    import (
    load_shapefile, match_districts, match_states,
    join_to_geo, join_state_to_geo,
    render_static, render_interactive,
    map_to_html_bytes, fig_to_bytes,
    detect_district_column,
)

# ── Brand colours ─────────────────────────────────────────────────────────────
C_BG        = "#0D0D0D"
C_SURFACE   = "#161616"
C_SURFACE2  = "#1E1E1E"
C_BORDER    = "#2A2A2A"
C_RED       = "#D42B2B"
C_RED_DIM   = "#8B1A1A"
C_TEXT      = "#F0F0F0"
C_TEXT_DIM  = "#888888"
C_TEXT_HINT = "#444444"
C_SUCCESS   = "#2ECC71"
C_WARNING   = "#F39C12"

BUNDLED_SHP = _HERE / "shapefiles" / "india_districts.zip"


# ══════════════════════════════════════════════════════════════════════════════
# Worker thread — runs the full agent pipeline off the main thread
# ══════════════════════════════════════════════════════════════════════════════

class PipelineWorker(QThread):
    progress  = Signal(str)          # status message
    profiled  = Signal(dict)         # profile_data result
    matched   = Signal(dict)         # match result summary
    rendered  = Signal(str, str)     # html_path, png_path
    error     = Signal(str)

    def __init__(self, file_path: str, geo_col: str, value_col: str,
                 level: str, title: str, cmap: str, n_classes: int):
        super().__init__()
        self.file_path = file_path
        self.geo_col   = geo_col
        self.value_col = value_col
        self.level     = level
        self.title     = title
        self.cmap      = cmap
        self.n_classes = n_classes

    def run(self):
        try:
            self.progress.emit("Loading shapefile…")
            gdf = load_shapefile(BUNDLED_SHP)

            self.progress.emit("Cleaning data…")
            df = clean_dataframe(self._read()).df

            geo_names = df[self.geo_col].astype(str).tolist()

            self.progress.emit("Matching geography names…")
            if self.level == "state":
                sc        = "STATE_UT" if "STATE_UT" in gdf.columns else gdf.columns[0]
                match_res = match_states(geo_names, gdf[sc].dropna().unique().tolist())
                merged    = join_state_to_geo(
                    gdf=gdf, data_df=df, data_name_col=self.geo_col,
                    state_col=sc, name_map=match_res.as_dict(),
                    value_col=self.value_col,
                )
                label_col = sc
            else:
                dc        = detect_district_column(gdf)
                match_res = match_districts(geo_names, gdf[dc].astype(str).tolist())
                merged    = join_to_geo(
                    gdf=gdf, data_df=df, data_name_col=self.geo_col,
                    shp_name_col=dc, name_map=match_res.as_dict(),
                    value_col=self.value_col,
                )
                label_col = dc

            self.matched.emit({
                "auto":      len(match_res.high_confidence),
                "review":    len(match_res.low_confidence),
                "unmatched": len(match_res.unmatched),
                "total":     len(geo_names),
            })

            self.progress.emit("Rendering interactive map…")
            fm = render_interactive(
                gdf=merged, value_col="_value", label_col=label_col,
                title=self.title, cmap_name=self.cmap, n_classes=self.n_classes,
            )
            tmp = tempfile.mkdtemp(prefix="mapgen_")
            html_path = str(Path(tmp) / "map.html")
            Path(html_path).write_bytes(map_to_html_bytes(fm))

            self.progress.emit("Rendering static PNG…")
            fig = render_static(
                merged, value_col="_value", scheme="quantiles",
                cmap_name=self.cmap, n_classes=self.n_classes, title=self.title,
            )
            png_path = str(Path(tmp) / "map.png")
            Path(png_path).write_bytes(fig_to_bytes(fig, "png"))

            self.progress.emit("Done.")
            self.rendered.emit(html_path, png_path)

        except Exception as exc:
            self.error.emit(str(exc))

    def _read(self) -> pd.DataFrame:
        p = Path(self.file_path)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        xl = pd.ExcelFile(p, engine="openpyxl")
        best, size = None, 0
        for s in xl.sheet_names:
            try:
                d = xl.parse(s)
                if d.size > size:
                    best, size = d, d.size
            except Exception:
                pass
        return best


class AnalysisWorker(QThread):
    done  = Signal(dict)
    error = Signal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            p = Path(self.file_path)
            if p.suffix.lower() == ".csv":
                raw = pd.read_csv(p)
            else:
                xl = pd.ExcelFile(p, engine="openpyxl")
                best, size = None, 0
                for s in xl.sheet_names:
                    try:
                        d = xl.parse(s)
                        if d.size > size:
                            best, size = d, d.size
                    except Exception:
                        pass
                raw = best

            cr      = clean_dataframe(raw)
            df      = cr.df
            profile = profile_data(df)
            gdf     = load_shapefile(BUNDLED_SHP)
            geo_names = df[profile.geo_col].astype(str).tolist() if profile.geo_col else []
            agent   = agent2_run(geo_names=geo_names, gdf=gdf) if geo_names else None
            emojis  = suggest_emoji(col_name=profile.geo_col or "")

            self.done.emit({
                "df":          df,
                "profile":     profile,
                "agent":       agent,
                "emojis":      emojis,
                "obj_cols":    list(df.select_dtypes("object").columns),
                "num_cols":    list(df.select_dtypes("number").columns),
            })
        except Exception as exc:
            self.error.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# Reusable widgets
# ══════════════════════════════════════════════════════════════════════════════

def _btn(text: str, primary: bool = False, small: bool = False) -> QPushButton:
    b = QPushButton(text)
    size = "0.78rem" if small else "0.88rem"
    pad  = "6px 14px" if small else "10px 24px"
    if primary:
        b.setStyleSheet(f"""
            QPushButton {{
                background: {C_RED}; color: #fff;
                border: none; border-radius: 6px;
                font-size: 13px; font-weight: 600;
                padding: {pad}; letter-spacing: 0.3px;
            }}
            QPushButton:hover {{ background: #E83535; }}
            QPushButton:pressed {{ background: {C_RED_DIM}; }}
            QPushButton:disabled {{ background: #333; color: #666; }}
        """)
    else:
        b.setStyleSheet(f"""
            QPushButton {{
                background: {C_SURFACE2}; color: {C_TEXT};
                border: 1px solid {C_BORDER}; border-radius: 6px;
                font-size: 13px; padding: {pad};
            }}
            QPushButton:hover {{ border-color: {C_RED}; color: {C_RED}; }}
            QPushButton:pressed {{ background: #222; }}
            QPushButton:disabled {{ color: #555; border-color: #222; }}
        """)
    return b


def _label(text: str, size: int = 13, dim: bool = False,
           bold: bool = False, color: str = "") -> QLabel:
    lbl = QLabel(text)
    c   = color or (C_TEXT_DIM if dim else C_TEXT)
    w   = "600" if bold else "400"
    lbl.setStyleSheet(f"color: {c}; font-size: {size}px; font-weight: {w};")
    return lbl


def _divider() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"background: {C_BORDER}; border: none; max-height: 1px;")
    return f


def _card(contents: QWidget | None = None) -> QWidget:
    w = QWidget()
    w.setStyleSheet(f"""
        QWidget {{
            background: {C_SURFACE};
            border: 1px solid {C_BORDER};
            border-radius: 10px;
        }}
    """)
    if contents:
        lay = QVBoxLayout(w)
        lay.addWidget(contents)
    return w


# ══════════════════════════════════════════════════════════════════════════════
# Drop Zone
# ══════════════════════════════════════════════════════════════════════════════

class DropZone(QWidget):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(180)
        self._hover = False

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(10)

        self._icon = _label("⊕", 40, color=C_RED)
        self._icon.setAlignment(Qt.AlignCenter)

        self._title = _label("Drop your data file here", 17, bold=True)
        self._title.setAlignment(Qt.AlignCenter)

        self._sub = _label("CSV or Excel · state-level or district-level · wide or long format",
                           12, dim=True)
        self._sub.setAlignment(Qt.AlignCenter)

        browse_btn = _btn("Browse files", small=True)
        browse_btn.setFixedWidth(120)
        browse_btn.clicked.connect(self._browse)

        lay.addWidget(self._icon)
        lay.addWidget(self._title)
        lay.addWidget(self._sub)
        lay.addSpacing(8)
        lay.addWidget(browse_btn, alignment=Qt.AlignCenter)

        self._update_style()

    def _update_style(self):
        border_color = C_RED if self._hover else C_BORDER
        bg           = "#1A1010" if self._hover else C_SURFACE
        self.setStyleSheet(f"""
            DropZone {{
                background: {bg};
                border: 2px dashed {border_color};
                border-radius: 12px;
            }}
        """)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            self._hover = True
            self._update_style()
            e.acceptProposedAction()

    def dragLeaveEvent(self, _):
        self._hover = False
        self._update_style()

    def dropEvent(self, e: QDropEvent):
        self._hover = False
        self._update_style()
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path.endswith((".csv", ".xlsx", ".xls")):
                self.file_dropped.emit(path)
                return

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open data file", str(Path.home()),
            "Data files (*.csv *.xlsx *.xls)"
        )
        if path:
            self.file_dropped.emit(path)


# ══════════════════════════════════════════════════════════════════════════════
# Stat chip
# ══════════════════════════════════════════════════════════════════════════════

class StatChip(QWidget):
    def __init__(self, label: str, value: str, accent: bool = False):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)

        val_lbl = _label(value, 22, bold=True, color=C_RED if accent else C_TEXT)
        key_lbl = _label(label, 11, dim=True)
        val_lbl.setAlignment(Qt.AlignCenter)
        key_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(val_lbl)
        lay.addWidget(key_lbl)

        self.setStyleSheet(f"""
            StatChip {{
                background: {C_SURFACE2};
                border: 1px solid {C_BORDER};
                border-radius: 8px;
            }}
        """)
        self.setMinimumWidth(90)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

class Sidebar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(200)
        self.setStyleSheet(f"background: {C_SURFACE}; border-right: 1px solid {C_BORDER};")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Wordmark
        header = QWidget()
        header.setStyleSheet(f"background: {C_BG}; border-bottom: 1px solid {C_BORDER};")
        hlay = QVBoxLayout(header)
        hlay.setContentsMargins(20, 20, 20, 16)
        hlay.setSpacing(2)
        wm = _label("mapgen", 20, bold=True, color=C_RED)
        sub = _label("Pahle India Foundation", 10, dim=True)
        hlay.addWidget(wm)
        hlay.addWidget(sub)
        lay.addWidget(header)

        # Steps
        self._step_widgets: list[QWidget] = []
        steps = [
            ("1", "Load Data"),
            ("2", "Configure"),
            ("3", "Match"),
            ("4", "Map"),
        ]
        steps_container = QWidget()
        slay = QVBoxLayout(steps_container)
        slay.setContentsMargins(12, 16, 12, 0)
        slay.setSpacing(4)

        for num, name in steps:
            row = QWidget()
            rlay = QHBoxLayout(row)
            rlay.setContentsMargins(10, 8, 10, 8)
            rlay.setSpacing(10)
            num_lbl = _label(num, 11, dim=True)
            num_lbl.setFixedWidth(14)
            num_lbl.setAlignment(Qt.AlignCenter)
            name_lbl = _label(name, 13, dim=True)
            rlay.addWidget(num_lbl)
            rlay.addWidget(name_lbl)
            row.setStyleSheet("border-radius: 6px;")
            slay.addWidget(row)
            self._step_widgets.append((row, num_lbl, name_lbl))

        lay.addWidget(steps_container)
        lay.addStretch()

        # Version
        ver = _label("v1.0  ·  mapgen(ai)", 10, dim=True)
        ver.setContentsMargins(20, 0, 0, 16)
        lay.addWidget(ver)

    def set_step(self, active: int):
        for i, (row, num_lbl, name_lbl) in enumerate(self._step_widgets):
            if i < active:
                row.setStyleSheet(f"border-radius:6px; background: transparent;")
                num_lbl.setStyleSheet(f"color: {C_RED}; font-size:11px;")
                name_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size:13px;")
            elif i == active:
                row.setStyleSheet(f"border-radius:6px; background: #1F1010;")
                num_lbl.setStyleSheet(f"color: {C_RED}; font-size:11px; font-weight:700;")
                name_lbl.setStyleSheet(f"color: {C_TEXT}; font-size:13px; font-weight:600;")
            else:
                row.setStyleSheet("border-radius:6px;")
                num_lbl.setStyleSheet(f"color: {C_TEXT_HINT}; font-size:11px;")
                name_lbl.setStyleSheet(f"color: {C_TEXT_HINT}; font-size:13px;")


# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════

class LoadPage(QWidget):
    """Page 0 — drop file + prompt."""
    file_ready = Signal(str, str)   # file_path, prompt

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {C_BG};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(48, 48, 48, 48)
        lay.setSpacing(24)

        # Hero header
        hero = _label("What do you want to map?", 28, bold=True)
        lay.addWidget(hero)

        # Prompt
        self._prompt = QTextEdit()
        self._prompt.setPlaceholderText(
            "Describe your data and goal — e.g.\n"
            "\"District-wise literacy rates for Rajasthan, highlight below-average districts.\"\n"
            "\"State-level GVA for 2024-25, show economic disparity across India.\""
        )
        self._prompt.setFixedHeight(100)
        self._prompt.setStyleSheet(f"""
            QTextEdit {{
                background: {C_SURFACE};
                border: 1px solid {C_BORDER};
                border-radius: 8px;
                color: {C_TEXT};
                font-size: 13px;
                padding: 12px;
                line-height: 1.5;
            }}
            QTextEdit:focus {{ border-color: {C_RED}; }}
        """)
        lay.addWidget(self._prompt)

        lay.addWidget(_divider())

        # Drop zone
        self._drop = DropZone()
        self._drop.file_dropped.connect(self._on_file)
        lay.addWidget(self._drop)

        lay.addStretch()

        # File badge (shown after drop)
        self._badge = QWidget()
        self._badge.setVisible(False)
        blay = QHBoxLayout(self._badge)
        blay.setContentsMargins(0, 0, 0, 0)
        self._badge_lbl = _label("", 13, color=C_SUCCESS)
        self._badge_lbl.setStyleSheet(f"""
            color: {C_SUCCESS};
            background: #0D2010;
            border: 1px solid #1A4020;
            border-radius: 6px;
            padding: 8px 14px;
            font-size: 13px;
        """)
        blay.addWidget(self._badge_lbl)
        blay.addStretch()
        lay.addWidget(self._badge)

        self._continue = _btn("Analyse data →", primary=True)
        self._continue.setEnabled(False)
        self._continue.setFixedWidth(160)
        self._continue.clicked.connect(self._emit)
        lay.addWidget(self._continue)

        self._file_path = ""

    def _on_file(self, path: str):
        self._file_path = path
        name = Path(path).name
        self._badge_lbl.setText(f"✓  {name}")
        self._badge.setVisible(True)
        self._continue.setEnabled(True)

    def _emit(self):
        self.file_ready.emit(self._file_path, self._prompt.toPlainText())


class ConfigPage(QWidget):
    """Page 1 — shows profile + lets user pick value col + title."""
    confirmed = Signal(dict)   # config dict

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {C_BG};")
        self._data: dict = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(48, 48, 48, 48)
        lay.setSpacing(20)

        lay.addWidget(_label("Data profile", 22, bold=True))

        # Stats row
        self._stats_row = QHBoxLayout()
        self._stats_row.setSpacing(10)
        lay.addLayout(self._stats_row)

        lay.addWidget(_divider())

        # Selectors
        grid = QVBoxLayout()
        grid.setSpacing(14)

        grid.addWidget(_label("Geography column", 12, dim=True))
        self._geo_combo = self._combo()
        grid.addWidget(self._geo_combo)

        grid.addWidget(_label("Value to map", 12, dim=True))
        self._val_combo = self._combo()
        grid.addWidget(self._val_combo)

        grid.addWidget(_label("Map title", 12, dim=True))
        self._title_input = QTextEdit()
        self._title_input.setFixedHeight(44)
        self._title_input.setStyleSheet(self._input_style())
        grid.addWidget(self._title_input)

        grid.addWidget(_label("Colour ramp", 12, dim=True))
        self._cmap_combo = self._combo()
        for c in ["YlOrRd", "Blues", "Greens", "RdYlGn", "Purples",
                  "OrRd", "BuPu", "YlGnBu", "plasma", "viridis"]:
            self._cmap_combo.addItem(c)
        grid.addWidget(self._cmap_combo)

        lay.addLayout(grid)

        lay.addStretch()

        # Notes
        self._notes = _label("", 12, dim=True)
        self._notes.setWordWrap(True)
        lay.addWidget(self._notes)

        row = QHBoxLayout()
        self._back = _btn("← Back")
        self._next = _btn("Run matching →", primary=True)
        self._next.clicked.connect(self._emit)
        row.addWidget(self._back)
        row.addStretch()
        row.addWidget(self._next)
        lay.addLayout(row)

    def _combo(self) -> QComboBox:
        c = QComboBox()
        c.setStyleSheet(f"""
            QComboBox {{
                background: {C_SURFACE};
                border: 1px solid {C_BORDER};
                border-radius: 6px;
                color: {C_TEXT};
                font-size: 13px;
                padding: 8px 12px;
            }}
            QComboBox:hover {{ border-color: {C_RED}; }}
            QComboBox QAbstractItemView {{
                background: {C_SURFACE2};
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                selection-background-color: {C_RED_DIM};
            }}
        """)
        return c

    def _input_style(self) -> str:
        return f"""
            QTextEdit {{
                background: {C_SURFACE};
                border: 1px solid {C_BORDER};
                border-radius: 6px;
                color: {C_TEXT};
                font-size: 13px;
                padding: 8px 12px;
            }}
            QTextEdit:focus {{ border-color: {C_RED}; }}
        """

    def populate(self, data: dict):
        self._data = data
        profile    = data["profile"]
        agent      = data["agent"]

        # Clear stats
        while self._stats_row.count():
            w = self._stats_row.takeAt(0)
            if w.widget():
                w.widget().deleteLater()

        level = agent.level if agent else profile.level
        self._stats_row.addWidget(StatChip("Rows", f"{profile.n_rows:,}"))
        self._stats_row.addWidget(StatChip("Format", profile.fmt.title()))
        self._stats_row.addWidget(StatChip("Level", level.title(), accent=True))
        if agent:
            self._stats_row.addWidget(StatChip("Matched",
                f"{agent.matched}/{agent.total}", accent=agent.matched == agent.total))
        self._stats_row.addStretch()

        # Geo combo
        self._geo_combo.clear()
        for c in data["obj_cols"]:
            self._geo_combo.addItem(c)
        if profile.geo_col in data["obj_cols"]:
            self._geo_combo.setCurrentText(profile.geo_col)

        # Value combo
        self._val_combo.clear()
        for c in data["num_cols"]:
            self._val_combo.addItem(c)
        if profile.value_cols:
            self._val_combo.setCurrentText(profile.value_cols[0])

        # Title
        emoji = data["emojis"][0][0] if data["emojis"] else "📊"
        suggested = f"{emoji} {self._val_combo.currentText().replace('_', ' ').title()} by {level.title()}"
        self._title_input.setPlainText(suggested)

        # Notes
        notes = " · ".join(profile.notes + profile.issues)
        self._notes.setText(notes)

    def _emit(self):
        self.confirmed.emit({
            "geo_col":   self._geo_combo.currentText(),
            "value_col": self._val_combo.currentText(),
            "level":     self._data.get("agent").level if self._data.get("agent") else "district",
            "title":     self._title_input.toPlainText().strip(),
            "cmap":      self._cmap_combo.currentText(),
            "n_classes": 5,
        })


class ProcessingPage(QWidget):
    """Page 2 — spinning progress while pipeline runs."""
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {C_BG};")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(16)

        self._spinner_angle = 0
        self._spinner = QLabel()
        self._spinner.setFixedSize(60, 60)
        self._spinner.setAlignment(Qt.AlignCenter)

        self._status = _label("Initialising…", 15, dim=True)
        self._status.setAlignment(Qt.AlignCenter)

        self._match_info = _label("", 13, color=C_SUCCESS)
        self._match_info.setAlignment(Qt.AlignCenter)
        self._match_info.setVisible(False)

        lay.addWidget(self._spinner)
        lay.addWidget(self._status)
        lay.addWidget(self._match_info)

        # Spinner timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._spin)
        self._timer.start(40)

    def _spin(self):
        self._spinner_angle = (self._spinner_angle + 8) % 360
        self._spinner.setText(["◐", "◓", "◑", "◒"][
            (self._spinner_angle // 90) % 4
        ])
        self._spinner.setStyleSheet(f"color: {C_RED}; font-size: 40px;")

    def set_status(self, msg: str):
        self._status.setText(msg)

    def set_match(self, d: dict):
        self._match_info.setText(
            f"✓  {d['auto']}/{d['total']} names matched  ·  "
            f"{d['unmatched']} unmatched"
        )
        self._match_info.setVisible(True)


class MapPage(QWidget):
    """Page 3 — full-bleed interactive map + export bar."""
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {C_BG};")
        self._html_path = ""
        self._png_path  = ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Top bar
        topbar = QWidget()
        topbar.setFixedHeight(52)
        topbar.setStyleSheet(f"background: {C_SURFACE}; border-bottom: 1px solid {C_BORDER};")
        tlay = QHBoxLayout(topbar)
        tlay.setContentsMargins(20, 0, 20, 0)

        self._map_title = _label("", 14, bold=True)
        self._match_badge = _label("", 12, dim=True)

        tlay.addWidget(self._map_title)
        tlay.addWidget(self._match_badge)
        tlay.addStretch()

        # Export buttons
        for label, slot in [
            ("PNG",  self._export_png),
            ("HTML", self._export_html),
            ("SVG",  self._export_svg),
            ("GeoJSON", self._export_geojson),
        ]:
            b = _btn(label, small=True)
            b.clicked.connect(slot)
            tlay.addWidget(b)

        tlay.addSpacing(8)
        self._new_btn = _btn("New map", primary=True, small=True)
        tlay.addWidget(self._new_btn)

        lay.addWidget(topbar)

        # Map view
        if HAS_WEBENGINE:
            self._view = QWebEngineView()
            self._view.setStyleSheet("background: #1a1a1a;")
            lay.addWidget(self._view)
        else:
            fallback = _label(
                "⚠  QtWebEngine not available.\n"
                "Map saved — open via Export → HTML.",
                16, dim=True
            )
            fallback.setAlignment(Qt.AlignCenter)
            lay.addWidget(fallback)
            self._view = None

    def load_map(self, html_path: str, png_path: str, title: str, match: dict):
        self._html_path = html_path
        self._png_path  = png_path
        self._map_title.setText(title)
        self._match_badge.setText(
            f"{match['auto']}/{match['total']} matched  ·  {match['unmatched']} unmatched"
        )
        if self._view and html_path:
            self._view.load(QUrl.fromLocalFile(html_path))

    def _export_png(self):
        if not self._png_path:
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save PNG", str(Path.home() / "Desktop" / "map.png"), "PNG (*.png)")
        if dest:
            import shutil; shutil.copy2(self._png_path, dest)

    def _export_html(self):
        if not self._html_path:
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save HTML", str(Path.home() / "Desktop" / "map.html"), "HTML (*.html)")
        if dest:
            import shutil; shutil.copy2(self._html_path, dest)

    def _export_svg(self):
        self._match_badge.setText("SVG — re-render needed (coming soon)")

    def _export_geojson(self):
        self._match_badge.setText("GeoJSON — available in web app")


# ══════════════════════════════════════════════════════════════════════════════
# Main window
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mapgen(ai)")
        self.resize(1280, 820)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(f"QMainWindow {{ background: {C_BG}; }}")

        self._file_path  = ""
        self._analysis   = {}
        self._match_data = {}
        self._worker: QThread | None = None

        # ── Root layout ───────────────────────────────────────────────────
        root = QWidget()
        root.setStyleSheet(f"background: {C_BG};")
        self.setCentralWidget(root)

        hlay = QHBoxLayout(root)
        hlay.setContentsMargins(0, 0, 0, 0)
        hlay.setSpacing(0)

        # Sidebar
        self._sidebar = Sidebar()
        hlay.addWidget(self._sidebar)

        # Pages
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background: {C_BG};")
        hlay.addWidget(self._stack)

        self._p_load       = LoadPage()
        self._p_config     = ConfigPage()
        self._p_processing = ProcessingPage()
        self._p_map        = MapPage()

        self._stack.addWidget(self._p_load)       # 0
        self._stack.addWidget(self._p_config)     # 1
        self._stack.addWidget(self._p_processing) # 2
        self._stack.addWidget(self._p_map)        # 3

        # ── Wire signals ──────────────────────────────────────────────────
        self._p_load.file_ready.connect(self._on_file_ready)
        self._p_config._back.clicked.connect(lambda: self._goto(0))
        self._p_config.confirmed.connect(self._on_config_confirmed)
        self._p_map._new_btn.clicked.connect(lambda: self._goto(0))

        self._goto(0)

    def _goto(self, idx: int):
        self._stack.setCurrentIndex(idx)
        self._sidebar.set_step(idx)

    def _on_file_ready(self, file_path: str, prompt: str):
        self._file_path = file_path
        self._goto(2)
        self._p_processing.set_status("Analysing data…")

        self._analysis_worker = AnalysisWorker(file_path)
        self._analysis_worker.done.connect(self._on_analysis_done)
        self._analysis_worker.error.connect(self._on_error)
        self._analysis_worker.start()

    def _on_analysis_done(self, data: dict):
        self._analysis = data
        self._p_config.populate(data)
        self._goto(1)

    def _on_config_confirmed(self, cfg: dict):
        self._cfg = cfg
        self._goto(2)
        self._p_processing.set_status("Loading shapefile…")
        self._p_processing._match_info.setVisible(False)

        self._pipeline = PipelineWorker(
            file_path = self._file_path,
            geo_col   = cfg["geo_col"],
            value_col = cfg["value_col"],
            level     = cfg["level"],
            title     = cfg["title"],
            cmap      = cfg["cmap"],
            n_classes = cfg["n_classes"],
        )
        self._pipeline.progress.connect(self._p_processing.set_status)
        self._pipeline.matched.connect(self._on_matched)
        self._pipeline.rendered.connect(self._on_rendered)
        self._pipeline.error.connect(self._on_error)
        self._pipeline.start()

    def _on_matched(self, d: dict):
        self._match_data = d
        self._p_processing.set_match(d)

    def _on_rendered(self, html_path: str, png_path: str):
        self._p_map.load_map(
            html_path, png_path,
            self._cfg["title"],
            self._match_data,
        )
        self._goto(3)

    def _on_error(self, msg: str):
        self._p_processing.set_status(f"⚠  Error: {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("mapgen(ai)")
    app.setOrganizationName("Pahle India Foundation")

    # macOS dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(C_BG))
    palette.setColor(QPalette.WindowText,      QColor(C_TEXT))
    palette.setColor(QPalette.Base,            QColor(C_SURFACE))
    palette.setColor(QPalette.AlternateBase,   QColor(C_SURFACE2))
    palette.setColor(QPalette.Text,            QColor(C_TEXT))
    palette.setColor(QPalette.Button,          QColor(C_SURFACE2))
    palette.setColor(QPalette.ButtonText,      QColor(C_TEXT))
    palette.setColor(QPalette.Highlight,       QColor(C_RED))
    palette.setColor(QPalette.HighlightedText, QColor("#fff"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
