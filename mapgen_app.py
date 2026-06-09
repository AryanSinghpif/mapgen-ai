"""
mapgen_app.py — Premium macOS desktop app for mapgen(ai)
=========================================================
PIF brand · Dark theme · Embedded interactive map · QWebEngineView
"""

from __future__ import annotations
import sys, os, shutil, tempfile
from pathlib import Path

import pandas as pd

from PySide6.QtCore  import (Qt, QThread, Signal, QTimer, QUrl,
                              QPropertyAnimation, QEasingCurve, QRect, QSize)
from PySide6.QtGui   import (QColor, QDragEnterEvent, QDropEvent, QFont,
                              QPalette, QLinearGradient, QPainter, QPen,
                              QBrush, QPixmap, QIcon, QFontMetrics)
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QProgressBar, QPushButton, QSizePolicy, QStackedWidget, QTextEdit,
    QVBoxLayout, QWidget, QComboBox, QFrame, QSplitter, QScrollArea,
    QGraphicsDropShadowEffect, QSpacerItem,
)
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from data_analyzer import clean_dataframe, profile_data, suggest_emoji
from state_agent   import run as agent2_run
from map_engine    import (
    load_shapefile, match_districts, match_states,
    join_to_geo, join_state_to_geo,
    render_static, render_interactive,
    map_to_html_bytes, fig_to_bytes,
    detect_district_column,
)

BUNDLED_SHP = _HERE / "shapefiles" / "india_districts.zip"

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#080808"
SURF      = "#101010"
SURF2     = "#161616"
SURF3     = "#1C1C1C"
BORDER    = "#242424"
BORDER2   = "#2E2E2E"
RED       = "#D42B2B"
RED2      = "#A82020"
RED_GLOW  = "rgba(212,43,43,0.15)"
RED_SOFT  = "rgba(212,43,43,0.08)"
TEXT      = "#EFEFEF"
TEXT2     = "#999999"
TEXT3     = "#555555"
SUCCESS   = "#34C759"
WARNING   = "#FF9F0A"
INFO      = "#0A84FF"

FONT_FAMILY = "-apple-system, 'SF Pro Display', 'Helvetica Neue', sans-serif"

GLOBAL_CSS = f"""
* {{
    font-family: -apple-system, 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif;
}}
QToolTip {{
    background: {SURF3};
    color: {TEXT};
    border: 1px solid {BORDER2};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER2};
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 6px; background: transparent; }}
QScrollBar::handle:horizontal {{ background: {BORDER2}; border-radius: 3px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def shadow(radius=24, color="#000000", opacity=0.5) -> QGraphicsDropShadowEffect:
    fx = QGraphicsDropShadowEffect()
    c  = QColor(color)
    c.setAlphaF(opacity)
    fx.setColor(c)
    fx.setBlurRadius(radius)
    fx.setOffset(0, 4)
    return fx

def btn(text, primary=False, danger=False, ghost=False, small=False) -> QPushButton:
    b  = QPushButton(text)
    fs = 12 if small else 13
    py = "6px" if small else "10px"
    px = "14px" if small else "20px"
    r  = "6px"
    if primary:
        css = f"""
        QPushButton {{
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 #E03030, stop:1 {RED2});
            color: #fff; border: none; border-radius: {r};
            font-size: {fs}px; font-weight: 600;
            padding: {py} {px}; letter-spacing: 0.2px;
        }}
        QPushButton:hover  {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #E83838, stop:1 #B52222); }}
        QPushButton:pressed{{ background: {RED2}; }}
        QPushButton:disabled{{ background: #282828; color: {TEXT3}; }}
        """
    elif ghost:
        css = f"""
        QPushButton {{
            background: transparent; color: {TEXT2};
            border: 1px solid {BORDER2}; border-radius: {r};
            font-size: {fs}px; padding: {py} {px};
        }}
        QPushButton:hover {{ border-color: {RED}; color: {TEXT}; }}
        QPushButton:pressed{{ background: {RED_SOFT}; }}
        """
    else:
        css = f"""
        QPushButton {{
            background: {SURF2}; color: {TEXT};
            border: 1px solid {BORDER}; border-radius: {r};
            font-size: {fs}px; padding: {py} {px};
        }}
        QPushButton:hover {{ background: {SURF3}; border-color: {BORDER2}; }}
        QPushButton:pressed{{ background: {SURF}; }}
        QPushButton:disabled{{ color: {TEXT3}; border-color: {BORDER}; }}
        """
    b.setStyleSheet(css)
    return b

def lbl(text, size=13, weight=400, color=None, wrap=False) -> QLabel:
    l = QLabel(text)
    c = color or TEXT
    l.setStyleSheet(f"color:{c}; font-size:{size}px; font-weight:{weight}; background:transparent;")
    l.setWordWrap(wrap)
    return l

def tag(text, color=RED) -> QLabel:
    l = QLabel(text)
    l.setStyleSheet(f"""
        color: {color}; background: transparent;
        border: 1px solid {color}; border-radius: 4px;
        font-size: 10px; font-weight: 600;
        padding: 2px 7px; letter-spacing: 0.5px;
    """)
    return l

def hdivider() -> QFrame:
    f = QFrame(); f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"background:{BORDER}; border:none; max-height:1px;")
    return f

def combo_style() -> str:
    return f"""
    QComboBox {{
        background: {SURF2}; color: {TEXT};
        border: 1px solid {BORDER}; border-radius: 7px;
        font-size: 13px; padding: 9px 12px;
    }}
    QComboBox:hover {{ border-color: {RED}; }}
    QComboBox::drop-down {{ border: none; width: 28px; }}
    QComboBox::down-arrow {{
        image: none; border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {TEXT2}; margin-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background: {SURF3}; color: {TEXT};
        border: 1px solid {BORDER2}; border-radius: 6px;
        selection-background-color: {RED2};
        padding: 4px;
        outline: none;
    }}
    QComboBox QAbstractItemView::item {{ padding: 6px 10px; border-radius: 4px; }}
    """

def input_style() -> str:
    return f"""
    QTextEdit {{
        background: {SURF2}; color: {TEXT};
        border: 1px solid {BORDER}; border-radius: 7px;
        font-size: 13px; padding: 10px 12px; line-height: 1.5;
        selection-background-color: {RED2};
    }}
    QTextEdit:focus {{ border-color: {RED}; background: {SURF3}; }}
    """


# ══════════════════════════════════════════════════════════════════════════════
# Workers
# ══════════════════════════════════════════════════════════════════════════════

class AnalysisWorker(QThread):
    done  = Signal(dict)
    error = Signal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            raw     = self._read()
            cr      = clean_dataframe(raw)
            df      = cr.df
            profile = profile_data(df)
            gdf     = load_shapefile(BUNDLED_SHP)
            names   = df[profile.geo_col].astype(str).tolist() if profile.geo_col else []
            agent   = agent2_run(geo_names=names, gdf=gdf) if names else None
            emojis  = suggest_emoji(col_name=profile.geo_col or "")
            self.done.emit({"df": df, "profile": profile, "agent": agent,
                            "emojis": emojis,
                            "obj_cols": list(df.select_dtypes("object").columns),
                            "num_cols": list(df.select_dtypes("number").columns)})
        except Exception as e:
            self.error.emit(str(e))

    def _read(self):
        p = Path(self.file_path)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        xl = pd.ExcelFile(p, engine="openpyxl")
        best, size = None, 0
        for s in xl.sheet_names:
            try:
                d = xl.parse(s)
                if d.size > size: best, size = d, d.size
            except Exception: pass
        return best


class PipelineWorker(QThread):
    progress = Signal(str)
    matched  = Signal(dict)
    rendered = Signal(str, str)
    error    = Signal(str)

    def __init__(self, file_path, geo_col, value_col, level, title, cmap, n_classes):
        super().__init__()
        self.__dict__.update(locals())

    def run(self):
        try:
            self.progress.emit("Loading India shapefile…")
            gdf = load_shapefile(BUNDLED_SHP)

            self.progress.emit("Cleaning & normalising data…")
            df  = clean_dataframe(self._read()).df
            names = df[self.geo_col].astype(str).tolist()

            self.progress.emit("Running name matching…")
            if self.level == "state":
                sc        = "STATE_UT" if "STATE_UT" in gdf.columns else gdf.columns[0]
                match_res = match_states(names, gdf[sc].dropna().unique().tolist())
                merged    = join_state_to_geo(gdf=gdf, data_df=df,
                    data_name_col=self.geo_col, state_col=sc,
                    name_map=match_res.as_dict(), value_col=self.value_col)
                label_col = sc
            else:
                dc        = detect_district_column(gdf)
                match_res = match_districts(names, gdf[dc].astype(str).tolist())
                merged    = join_to_geo(gdf=gdf, data_df=df,
                    data_name_col=self.geo_col, shp_name_col=dc,
                    name_map=match_res.as_dict(), value_col=self.value_col)
                label_col = dc

            self.matched.emit({"auto": len(match_res.high_confidence),
                               "review": len(match_res.low_confidence),
                               "unmatched": len(match_res.unmatched),
                               "total": len(names)})

            self.progress.emit("Rendering interactive map…")
            fm  = render_interactive(gdf=merged, value_col="_value", label_col=label_col,
                                     title=self.title, cmap_name=self.cmap, n_classes=self.n_classes)
            tmp = tempfile.mkdtemp(prefix="mapgen_")
            hp  = str(Path(tmp) / "map.html")
            Path(hp).write_bytes(map_to_html_bytes(fm))

            self.progress.emit("Rendering static map…")
            fig = render_static(merged, value_col="_value", scheme="quantiles",
                                cmap_name=self.cmap, n_classes=self.n_classes, title=self.title)
            pp  = str(Path(tmp) / "map.png")
            Path(pp).write_bytes(fig_to_bytes(fig, "png"))

            self.progress.emit("Ready.")
            self.rendered.emit(hp, pp)
        except Exception as e:
            self.error.emit(str(e))

    def _read(self):
        p = Path(self.file_path)
        if p.suffix.lower() == ".csv": return pd.read_csv(p)
        xl = pd.ExcelFile(p, engine="openpyxl")
        best, size = None, 0
        for s in xl.sheet_names:
            try:
                d = xl.parse(s)
                if d.size > size: best, size = d, d.size
            except Exception: pass
        return best


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

class Sidebar(QWidget):
    STEPS = [("Load", "Upload your data"),
             ("Configure", "Choose columns"),
             ("Processing", "Running agents"),
             ("Map", "Interactive output")]

    def __init__(self):
        super().__init__()
        self.setFixedWidth(220)
        self.setStyleSheet(f"""
            Sidebar {{
                background: {SURF};
                border-right: 1px solid {BORDER};
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet(f"background: {BG}; border-bottom: 1px solid {BORDER};")
        header.setFixedHeight(72)
        hl = QVBoxLayout(header)
        hl.setContentsMargins(24, 16, 24, 16)
        hl.setSpacing(2)
        wm = lbl("mapgen", 20, 700, RED)
        sub = lbl("Pahle India Foundation", 10, 400, TEXT3)
        hl.addWidget(wm)
        hl.addWidget(sub)
        lay.addWidget(header)

        # ── Steps ───────────────────────────────────────────────────────
        sc = QWidget()
        sc.setStyleSheet("background: transparent;")
        sl = QVBoxLayout(sc)
        sl.setContentsMargins(16, 20, 16, 0)
        sl.setSpacing(2)

        self._rows = []
        for i, (name, desc) in enumerate(self.STEPS):
            row = QWidget()
            row.setStyleSheet("border-radius: 8px; background: transparent;")
            rl  = QHBoxLayout(row)
            rl.setContentsMargins(10, 10, 10, 10)
            rl.setSpacing(12)

            dot = QLabel("●")
            dot.setFixedWidth(10)
            dot.setAlignment(Qt.AlignCenter)

            info = QWidget()
            il   = QVBoxLayout(info)
            il.setContentsMargins(0, 0, 0, 0)
            il.setSpacing(1)
            n_lbl = lbl(name, 13, 500, TEXT3)
            d_lbl = lbl(desc, 11, 400, TEXT3)
            il.addWidget(n_lbl)
            il.addWidget(d_lbl)

            rl.addWidget(dot)
            rl.addWidget(info)
            sl.addWidget(row)
            self._rows.append((row, dot, n_lbl, d_lbl))

        lay.addWidget(sc)
        lay.addStretch()

        # ── Footer ──────────────────────────────────────────────────────
        ft = QWidget()
        ft.setStyleSheet(f"background: transparent; border-top: 1px solid {BORDER};")
        fl = QHBoxLayout(ft)
        fl.setContentsMargins(24, 12, 24, 14)
        fl.addWidget(lbl("mapgen(ai)  ·  v1.0", 10, 400, TEXT3))
        lay.addWidget(ft)

    def set_step(self, idx: int):
        for i, (row, dot, n, d) in enumerate(self._rows):
            if i < idx:
                row.setStyleSheet("border-radius:8px; background:transparent;")
                dot.setStyleSheet(f"color:{RED}; font-size:8px; background:transparent;")
                n.setStyleSheet(f"color:{TEXT2}; font-size:13px; font-weight:500; background:transparent;")
                d.setStyleSheet(f"color:{TEXT3}; font-size:11px; background:transparent;")
            elif i == idx:
                row.setStyleSheet(f"border-radius:8px; background:{RED_SOFT}; border: 1px solid {RED_GLOW};")
                dot.setStyleSheet(f"color:{RED}; font-size:10px; background:transparent;")
                n.setStyleSheet(f"color:{TEXT}; font-size:13px; font-weight:600; background:transparent;")
                d.setStyleSheet(f"color:{TEXT2}; font-size:11px; background:transparent;")
            else:
                row.setStyleSheet("border-radius:8px; background:transparent;")
                dot.setStyleSheet(f"color:{TEXT3}; font-size:8px; background:transparent;")
                n.setStyleSheet(f"color:{TEXT3}; font-size:13px; background:transparent;")
                d.setStyleSheet(f"color:{TEXT3}; font-size:11px; background:transparent;")


# ══════════════════════════════════════════════════════════════════════════════
# Page 0 — Load
# ══════════════════════════════════════════════════════════════════════════════

class DropZone(QWidget):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(200)
        self._hover = False

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(10)
        lay.setContentsMargins(40, 32, 40, 32)

        self._icon = lbl("⊕", 44, 300, RED)
        self._icon.setAlignment(Qt.AlignCenter)

        self._title = lbl("Drop your data file here", 16, 600, TEXT)
        self._title.setAlignment(Qt.AlignCenter)

        self._sub = lbl("CSV or Excel  ·  state or district  ·  wide or long format",
                        12, 400, TEXT2)
        self._sub.setAlignment(Qt.AlignCenter)

        browse = btn("Browse files →", ghost=True, small=True)
        browse.setFixedWidth(130)
        browse.clicked.connect(self._browse)

        lay.addWidget(self._icon)
        lay.addWidget(self._title)
        lay.addWidget(self._sub)
        lay.addSpacing(10)
        lay.addWidget(browse, alignment=Qt.AlignCenter)
        self._update_style()

    def _update_style(self):
        if self._hover:
            self.setStyleSheet(f"""
                DropZone {{
                    background: {RED_SOFT};
                    border: 2px dashed {RED};
                    border-radius: 14px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                DropZone {{
                    background: {SURF2};
                    border: 1.5px dashed {BORDER2};
                    border-radius: 14px;
                }}
            """)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            self._hover = True; self._update_style(); e.acceptProposedAction()

    def dragLeaveEvent(self, _):
        self._hover = False; self._update_style()

    def dropEvent(self, e):
        self._hover = False; self._update_style()
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if p.endswith((".csv", ".xlsx", ".xls")):
                self.file_dropped.emit(p); return

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open data file", str(Path.home()),
            "Data files (*.csv *.xlsx *.xls)")
        if p: self.file_dropped.emit(p)


class LoadPage(QWidget):
    file_ready = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background:{BG};")
        self._path = ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(56, 52, 56, 52)
        lay.setSpacing(0)

        # Hero
        lay.addWidget(lbl("What do you want to map?", 30, 700, TEXT))
        lay.addSpacing(6)
        lay.addWidget(lbl("Describe your data — the tool will figure out the rest.",
                          14, 400, TEXT2))
        lay.addSpacing(28)

        # Prompt
        self._prompt = QTextEdit()
        self._prompt.setPlaceholderText(
            "e.g.  \"District-wise literacy rates for Rajasthan, highlight below-average districts.\"\n"
            "or    \"All-India state GVA 2024-25, show economic disparity.\""
        )
        self._prompt.setFixedHeight(90)
        self._prompt.setStyleSheet(input_style())
        lay.addWidget(self._prompt)
        lay.addSpacing(28)

        # Drop zone
        self._drop = DropZone()
        self._drop.file_dropped.connect(self._on_file)
        lay.addWidget(self._drop)
        lay.addSpacing(20)

        # File badge
        self._badge = QLabel("")
        self._badge.setVisible(False)
        self._badge.setStyleSheet(f"""
            color: {SUCCESS}; background: rgba(52,199,89,0.08);
            border: 1px solid rgba(52,199,89,0.25);
            border-radius: 8px; font-size: 13px;
            padding: 10px 16px;
        """)
        lay.addWidget(self._badge)
        lay.addStretch()

        # CTA
        row = QHBoxLayout()
        self._cta = btn("Analyse data →", primary=True)
        self._cta.setEnabled(False)
        self._cta.setFixedWidth(160)
        self._cta.clicked.connect(lambda: self.file_ready.emit(self._path, self._prompt.toPlainText()))
        row.addStretch()
        row.addWidget(self._cta)
        lay.addLayout(row)

    def _on_file(self, p):
        self._path = p
        self._badge.setText(f"  ✓   {Path(p).name}  ·  {Path(p).suffix[1:].upper()}")
        self._badge.setVisible(True)
        self._cta.setEnabled(True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 1 — Configure
# ══════════════════════════════════════════════════════════════════════════════

class StatCard(QWidget):
    def __init__(self, label, value, accent=False):
        super().__init__()
        self.setStyleSheet(f"""
            StatCard {{
                background: {SURF2};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(4)
        v = lbl(str(value), 24, 700, RED if accent else TEXT)
        v.setAlignment(Qt.AlignLeft)
        k = lbl(label.upper(), 10, 500, TEXT3)
        k.setAlignment(Qt.AlignLeft)
        k.setStyleSheet(f"color:{TEXT3}; font-size:10px; font-weight:500; letter-spacing:1px; background:transparent;")
        lay.addWidget(v)
        lay.addWidget(k)
        self.setMinimumWidth(100)


class ConfigPage(QWidget):
    confirmed = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background:{BG};")
        self._data = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Top header bar ───────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(64)
        header.setStyleSheet(f"background:{SURF}; border-bottom:1px solid {BORDER};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(32, 0, 32, 0)
        hl.addWidget(lbl("Configure your map", 16, 600, TEXT))
        hl.addStretch()
        self._back = btn("← Back", ghost=True, small=True)
        hl.addWidget(self._back)
        outer.addWidget(header)

        # ── Stats row ────────────────────────────────────────────────────
        self._stats_bar = QWidget()
        self._stats_bar.setStyleSheet(f"background:{BG};")
        self._stats_lay = QHBoxLayout(self._stats_bar)
        self._stats_lay.setContentsMargins(32, 24, 32, 0)
        self._stats_lay.setSpacing(12)
        outer.addWidget(self._stats_bar)

        # ── Scroll content ───────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        content = QWidget()
        content.setStyleSheet(f"background:{BG};")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(32, 24, 32, 32)
        cl.setSpacing(20)

        # Two-column layout
        cols = QHBoxLayout()
        cols.setSpacing(20)

        left  = QVBoxLayout(); left.setSpacing(16)
        right = QVBoxLayout(); right.setSpacing(16)

        # Geo col
        left.addWidget(self._field_label("Geography column"))
        self._geo = QComboBox(); self._geo.setStyleSheet(combo_style())
        left.addWidget(self._geo)

        # Value col
        left.addWidget(self._field_label("Value to map"))
        self._val = QComboBox(); self._val.setStyleSheet(combo_style())
        left.addWidget(self._val)

        # Title
        left.addWidget(self._field_label("Map title"))
        self._title_input = QTextEdit()
        self._title_input.setFixedHeight(52)
        self._title_input.setStyleSheet(input_style())
        left.addWidget(self._title_input)
        left.addStretch()

        # Colour ramp
        right.addWidget(self._field_label("Colour ramp"))
        self._cmap = QComboBox(); self._cmap.setStyleSheet(combo_style())
        for c in ["YlOrRd","Blues","Greens","RdYlGn","Purples",
                  "OrRd","BuPu","YlGnBu","plasma","viridis","magma"]:
            self._cmap.addItem(c)
        right.addWidget(self._cmap)

        # Level tag (read-only)
        right.addWidget(self._field_label("Detected level"))
        self._level_lbl = QLabel("—")
        self._level_lbl.setStyleSheet(f"""
            color:{TEXT}; background:{SURF2};
            border:1px solid {BORDER}; border-radius:7px;
            font-size:13px; padding: 9px 12px;
        """)
        right.addWidget(self._level_lbl)

        # Notes
        right.addWidget(self._field_label("Agent notes"))
        self._notes = QLabel("")
        self._notes.setWordWrap(True)
        self._notes.setStyleSheet(f"""
            color:{TEXT2}; background:{SURF2};
            border:1px solid {BORDER}; border-radius:7px;
            font-size:12px; padding: 10px 12px;
            line-height:1.5;
        """)
        self._notes.setMinimumHeight(80)
        right.addWidget(self._notes)
        right.addStretch()

        cols.addLayout(left,  1)
        cols.addLayout(right, 1)
        cl.addLayout(cols)

        # CTA
        cta_row = QHBoxLayout()
        self._run_btn = btn("Run matching & render →", primary=True)
        self._run_btn.setFixedWidth(220)
        self._run_btn.clicked.connect(self._emit)
        cta_row.addStretch()
        cta_row.addWidget(self._run_btn)
        cl.addLayout(cta_row)

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _field_label(self, text) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet(f"color:{TEXT3}; font-size:11px; font-weight:500; "
                        f"letter-spacing:0.8px; background:transparent;")
        return l

    def populate(self, data: dict):
        self._data = data
        profile = data["profile"]
        agent   = data["agent"]

        # Stats
        while self._stats_lay.count():
            w = self._stats_lay.takeAt(0)
            if w.widget(): w.widget().deleteLater()

        lvl = agent.level if agent else profile.level
        self._stats_lay.addWidget(StatCard("Rows",    f"{profile.n_rows:,}"))
        self._stats_lay.addWidget(StatCard("Columns", f"{profile.n_cols}"))
        self._stats_lay.addWidget(StatCard("Format",  profile.fmt.title()))
        self._stats_lay.addWidget(StatCard("Level",   lvl.title(), accent=True))
        if agent:
            pct = f"{agent.matched/max(agent.total,1):.0%}"
            self._stats_lay.addWidget(StatCard("Matched", pct, accent=True))
        self._stats_lay.addStretch()

        # Combos
        self._geo.clear(); [self._geo.addItem(c) for c in data["obj_cols"]]
        if profile.geo_col in data["obj_cols"]:
            self._geo.setCurrentText(profile.geo_col)

        self._val.clear(); [self._val.addItem(c) for c in data["num_cols"]]
        if profile.value_cols:
            self._val.setCurrentText(profile.value_cols[0])

        # Title
        emoji = data["emojis"][0][0] if data["emojis"] else "📊"
        self._title_input.setPlainText(
            f"{emoji} {self._val.currentText().replace('_',' ').title()} by {lvl.title()}"
        )

        # Level
        conf = f"{agent.level_confidence:.0%}" if agent else "?"
        self._level_lbl.setText(f"{lvl.title()}  ·  {conf} confidence")

        # Notes
        notes = profile.notes + profile.issues
        self._notes.setText("\n".join(f"· {n}" for n in notes) if notes else "No issues detected.")

    def _emit(self):
        agent = self._data.get("agent")
        self.confirmed.emit({
            "geo_col":   self._geo.currentText(),
            "value_col": self._val.currentText(),
            "level":     agent.level if agent else "district",
            "title":     self._title_input.toPlainText().strip(),
            "cmap":      self._cmap.currentText(),
            "n_classes": 5,
        })


# ══════════════════════════════════════════════════════════════════════════════
# Page 2 — Processing
# ══════════════════════════════════════════════════════════════════════════════

class ProcessingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background:{BG};")
        self._angle  = 0
        self._frames = ["◜◝", "◝◞", "◞◟", "◟◜"]
        self._fi     = 0

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(0)

        # Big spinner
        self._spin = QLabel("◐")
        self._spin.setAlignment(Qt.AlignCenter)
        self._spin.setStyleSheet(f"color:{RED}; font-size:56px; background:transparent;")
        self._spin.setFixedHeight(80)
        lay.addWidget(self._spin)
        lay.addSpacing(28)

        # Status
        self._status = lbl("Initialising…", 15, 400, TEXT2)
        self._status.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._status)
        lay.addSpacing(12)

        # Sub-status
        self._sub = lbl("", 12, 400, TEXT3)
        self._sub.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._sub)
        lay.addSpacing(32)

        # Match badge
        self._match = QLabel("")
        self._match.setAlignment(Qt.AlignCenter)
        self._match.setStyleSheet(f"""
            color:{SUCCESS}; background:rgba(52,199,89,0.08);
            border:1px solid rgba(52,199,89,0.2); border-radius:8px;
            font-size:13px; padding:10px 20px;
        """)
        self._match.setVisible(False)
        lay.addWidget(self._match, alignment=Qt.AlignCenter)

        # Step pipeline display
        lay.addSpacing(40)
        self._steps_row = QHBoxLayout()
        self._steps_row.setAlignment(Qt.AlignCenter)
        self._steps_row.setSpacing(8)
        self._step_lbls = []
        steps = ["Load", "Clean", "Match", "Render"]
        for i, s in enumerate(steps):
            if i > 0:
                sep = lbl("→", 12, 400, TEXT3)
                sep.setAlignment(Qt.AlignCenter)
                self._steps_row.addWidget(sep)
            sl = lbl(s, 12, 500, TEXT3)
            sl.setAlignment(Qt.AlignCenter)
            sl.setStyleSheet(f"""
                color:{TEXT3}; background:{SURF2};
                border:1px solid {BORDER}; border-radius:6px;
                font-size:12px; font-weight:500;
                padding:5px 14px;
            """)
            self._steps_row.addWidget(sl)
            self._step_lbls.append(sl)
        lay.addLayout(self._steps_row)

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(120)
        self._active_step = -1

    def _tick(self):
        frames = ["◐", "◓", "◑", "◒"]
        self._fi = (self._fi + 1) % 4
        self._spin.setText(frames[self._fi])

    def set_status(self, msg: str):
        self._status.setText(msg)
        step_map = {"Load": 0, "Clean": 1, "Match": 2, "Render": 3}
        for k, i in step_map.items():
            if k.lower() in msg.lower():
                self._activate_step(i)

    def _activate_step(self, idx: int):
        if idx == self._active_step: return
        self._active_step = idx
        for i, sl in enumerate(self._step_lbls):
            if i < idx:
                sl.setStyleSheet(f"color:{SUCCESS}; background:rgba(52,199,89,0.08); "
                                 f"border:1px solid rgba(52,199,89,0.2); border-radius:6px; "
                                 f"font-size:12px; font-weight:500; padding:5px 14px;")
            elif i == idx:
                sl.setStyleSheet(f"color:{TEXT}; background:{RED_SOFT}; "
                                 f"border:1px solid {RED}; border-radius:6px; "
                                 f"font-size:12px; font-weight:600; padding:5px 14px;")
            else:
                sl.setStyleSheet(f"color:{TEXT3}; background:{SURF2}; "
                                 f"border:1px solid {BORDER}; border-radius:6px; "
                                 f"font-size:12px; font-weight:500; padding:5px 14px;")

    def set_match(self, d: dict):
        self._match.setText(
            f"✓  {d['auto']}/{d['total']} names matched  ·  "
            f"{d['unmatched']} unmatched  ·  {d['review']} low-confidence"
        )
        self._match.setVisible(True)
        self._activate_step(3)

    def reset(self):
        self._match.setVisible(False)
        self._active_step = -1
        for sl in self._step_lbls:
            sl.setStyleSheet(f"color:{TEXT3}; background:{SURF2}; "
                             f"border:1px solid {BORDER}; border-radius:6px; "
                             f"font-size:12px; font-weight:500; padding:5px 14px;")


# ══════════════════════════════════════════════════════════════════════════════
# Page 3 — Map
# ══════════════════════════════════════════════════════════════════════════════

class MapPage(QWidget):
    new_map = Signal()

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background:{BG};")
        self._html = ""; self._png = ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Floating top bar ─────────────────────────────────────────────
        topbar = QWidget()
        topbar.setFixedHeight(56)
        topbar.setStyleSheet(f"""
            background: rgba(16,16,16,0.95);
            border-bottom: 1px solid {BORDER};
        """)
        tl = QHBoxLayout(topbar)
        tl.setContentsMargins(24, 0, 24, 0)
        tl.setSpacing(12)

        self._title_lbl = lbl("", 14, 600, TEXT)
        self._badge_lbl = lbl("", 12, 400, TEXT2)
        tl.addWidget(self._title_lbl)
        tl.addWidget(self._badge_lbl)
        tl.addStretch()

        # Export buttons
        for label, cb in [("PNG", self._export_png), ("HTML", self._export_html)]:
            b = btn(label, ghost=True, small=True)
            b.clicked.connect(cb)
            tl.addWidget(b)

        tl.addSpacing(4)
        nb = btn("↩  New map", primary=True, small=True)
        nb.clicked.connect(self.new_map.emit)
        tl.addWidget(nb)
        lay.addWidget(topbar)

        # ── Map view ─────────────────────────────────────────────────────
        if HAS_WEBENGINE:
            self._view = QWebEngineView()
            self._view.setStyleSheet("background:#111;")
            lay.addWidget(self._view)
        else:
            fl = lbl("⚠  Install PySide6-WebEngine to see map inline.\n"
                     "Use Export → HTML to open in browser.", 15, 400, TEXT2)
            fl.setAlignment(Qt.AlignCenter)
            lay.addWidget(fl)
            self._view = None

    def load(self, html: str, png: str, title: str, match: dict):
        self._html = html; self._png = png
        self._title_lbl.setText(title)
        pct = f"{match['auto']}/{match['total']} matched"
        um  = f"  ·  {match['unmatched']} unmatched" if match['unmatched'] else ""
        self._badge_lbl.setText(pct + um)
        if self._view:
            self._view.load(QUrl.fromLocalFile(html))

    def _export_png(self):
        if not self._png: return
        dst, _ = QFileDialog.getSaveFileName(
            self, "Save PNG", str(Path.home() / "Desktop" / "map.png"), "PNG (*.png)")
        if dst: shutil.copy2(self._png, dst)

    def _export_html(self):
        if not self._html: return
        dst, _ = QFileDialog.getSaveFileName(
            self, "Save HTML", str(Path.home() / "Desktop" / "map.html"), "HTML (*.html)")
        if dst: shutil.copy2(self._html, dst)


# ══════════════════════════════════════════════════════════════════════════════
# Main Window
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mapgen(ai)")
        self.resize(1300, 840)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(f"QMainWindow {{ background:{BG}; }}")

        self._file  = ""
        self._cfg   = {}
        self._match = {}

        root = QWidget(); root.setStyleSheet(f"background:{BG};")
        self.setCentralWidget(root)
        hl = QHBoxLayout(root); hl.setContentsMargins(0,0,0,0); hl.setSpacing(0)

        self._sidebar = Sidebar()
        hl.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background:{BG};")
        hl.addWidget(self._stack)

        self._p0 = LoadPage()
        self._p1 = ConfigPage()
        self._p2 = ProcessingPage()
        self._p3 = MapPage()
        for p in [self._p0, self._p1, self._p2, self._p3]:
            self._stack.addWidget(p)

        # Wire
        self._p0.file_ready.connect(self._on_file)
        self._p1._back.clicked.connect(lambda: self._go(0))
        self._p1.confirmed.connect(self._on_config)
        self._p3.new_map.connect(lambda: self._go(0))

        self._go(0)

    def _go(self, i):
        self._stack.setCurrentIndex(i)
        self._sidebar.set_step(i)

    def _on_file(self, path, prompt):
        self._file = path
        self._go(2)
        self._p2.reset()
        self._p2.set_status("Loading data…")
        self._aw = AnalysisWorker(path)
        self._aw.done.connect(lambda d: (self._p1.populate(d), self._go(1)))
        self._aw.error.connect(lambda e: self._p2.set_status(f"⚠ {e}"))
        self._aw.start()

    def _on_config(self, cfg):
        self._cfg = cfg
        self._go(2)
        self._p2.reset()
        self._p2.set_status("Loading India shapefile…")
        self._pw = PipelineWorker(self._file, cfg["geo_col"], cfg["value_col"],
                                  cfg["level"], cfg["title"], cfg["cmap"], cfg["n_classes"])
        self._pw.progress.connect(self._p2.set_status)
        self._pw.matched.connect(self._on_matched)
        self._pw.rendered.connect(self._on_rendered)
        self._pw.error.connect(lambda e: self._p2.set_status(f"⚠ {e}"))
        self._pw.start()

    def _on_matched(self, d):
        self._match = d
        self._p2.set_match(d)

    def _on_rendered(self, html, png):
        self._p3.load(html, png, self._cfg["title"], self._match)
        self._go(3)


# ══════════════════════════════════════════════════════════════════════════════
# Entry
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("mapgen(ai)")
    app.setStyleSheet(GLOBAL_CSS)

    pal = QPalette()
    pal.setColor(QPalette.Window,        QColor(BG))
    pal.setColor(QPalette.WindowText,    QColor(TEXT))
    pal.setColor(QPalette.Base,          QColor(SURF))
    pal.setColor(QPalette.Text,          QColor(TEXT))
    pal.setColor(QPalette.Button,        QColor(SURF2))
    pal.setColor(QPalette.ButtonText,    QColor(TEXT))
    pal.setColor(QPalette.Highlight,     QColor(RED))
    pal.setColor(QPalette.HighlightedText, QColor("#fff"))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
