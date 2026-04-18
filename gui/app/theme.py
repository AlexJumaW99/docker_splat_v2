"""
Theme: near-black canvas + electric-violet accent. All colors live here so
retheming is a single-file operation.
"""
from __future__ import annotations

from PySide6.QtGui import QPalette, QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

COLORS = {
    "canvas":     "#0E0F13",
    "surface":    "#171921",
    "surface_hi": "#1C1F2A",
    "border":     "#262933",
    "text":       "#E6E8EE",
    "text_dim":   "#9AA0AE",
    "accent":     "#7C5CFF",
    "accent_hi":  "#9B82FF",
    "success":    "#22C55E",
    "warning":    "#F59E0B",
    "error":      "#EF4444",
}


def _pick_font(candidates: list[str], fallback: str) -> str:
    families = set(QFontDatabase.families())
    for name in candidates:
        if name in families:
            return name
    return fallback


def apply_theme(app: QApplication) -> None:
    ui = _pick_font(["Inter", "Ubuntu", "Noto Sans"], app.font().family())
    mono = _pick_font(["JetBrains Mono", "Ubuntu Mono", "DejaVu Sans Mono"], "monospace")

    app.setFont(QFont(ui, 10))

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(COLORS["canvas"]))
    pal.setColor(QPalette.Base,            QColor(COLORS["surface"]))
    pal.setColor(QPalette.AlternateBase,   QColor(COLORS["surface_hi"]))
    pal.setColor(QPalette.Text,            QColor(COLORS["text"]))
    pal.setColor(QPalette.WindowText,      QColor(COLORS["text"]))
    pal.setColor(QPalette.Button,          QColor(COLORS["surface_hi"]))
    pal.setColor(QPalette.ButtonText,      QColor(COLORS["text"]))
    pal.setColor(QPalette.Highlight,       QColor(COLORS["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.ToolTipBase,     QColor(COLORS["surface_hi"]))
    pal.setColor(QPalette.ToolTipText,     QColor(COLORS["text"]))
    app.setPalette(pal)

    app.setStyleSheet(_QSS.format(mono=mono, **COLORS))


_QSS = """
* {{
    color: {text};
    font-size: 13px;
}}

QMainWindow, QWidget#Canvas, QDialog {{
    background-color: {canvas};
}}

QFrame#Sidebar {{
    background-color: {surface};
    border-right: 1px solid {border};
}}

QLabel#AppTitle {{
    font-size: 15px;
    font-weight: 600;
    padding: 14px 16px 14px 16px;
    color: {text};
}}

QPushButton[role="nav"] {{
    text-align: left;
    padding: 10px 18px;
    border: none;
    background: transparent;
    color: {text_dim};
    border-left: 2px solid transparent;
}}
QPushButton[role="nav"]:hover {{
    background-color: {surface_hi};
    color: {text};
}}
QPushButton[role="nav"][active="true"] {{
    color: {text};
    border-left: 2px solid {accent};
    background-color: {surface_hi};
}}

QLabel#PageTitle {{
    font-size: 22px;
    font-weight: 600;
    color: {text};
}}

QFrame[role="card"] {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 10px;
}}

QLabel[role="cardTitle"] {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    color: {text_dim};
}}

QLabel[role="cardValue"] {{
    font-size: 20px;
    font-weight: 600;
    color: {text};
}}

QLabel[role="metric"] {{
    font-size: 18px;
    font-weight: 600;
    color: {text};
}}

QLabel[role="caption"] {{
    color: {text_dim};
    font-size: 12px;
}}

QPushButton[role="primary"] {{
    background-color: {accent};
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}}
QPushButton[role="primary"]:hover  {{ background-color: {accent_hi}; }}
QPushButton[role="primary"]:disabled {{
    background-color: {border};
    color: {text_dim};
}}

QPushButton[role="secondary"] {{
    background-color: {surface_hi};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 8px 16px;
}}
QPushButton[role="secondary"]:hover {{ border-color: {accent}; }}

QLabel[pill="success"] {{
    background-color: rgba(34, 197, 94, 40);
    color: {success};
    border: 1px solid rgba(34, 197, 94, 100);
    border-radius: 10px;
    padding: 2px 10px;
    font-weight: 600;
}}
QLabel[pill="warning"] {{
    background-color: rgba(245, 158, 11, 40);
    color: {warning};
    border: 1px solid rgba(245, 158, 11, 100);
    border-radius: 10px;
    padding: 2px 10px;
    font-weight: 600;
}}
QLabel[pill="error"] {{
    background-color: rgba(239, 68, 68, 40);
    color: {error};
    border: 1px solid rgba(239, 68, 68, 100);
    border-radius: 10px;
    padding: 2px 10px;
    font-weight: 600;
}}
QLabel[pill="neutral"] {{
    background-color: {surface_hi};
    color: {text_dim};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 2px 10px;
}}

QPlainTextEdit, QTextEdit {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 6px;
    font-family: "{mono}";
    color: {text};
    padding: 6px;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {border};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {text_dim}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    height: 0; background: transparent;
}}

QToolTip {{
    background-color: {surface_hi};
    color: {text};
    border: 1px solid {border};
    padding: 4px 8px;
    border-radius: 4px;
}}

/* ---------- forms ---------- */

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {surface_hi};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {accent};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {accent};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {text_dim};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {text_dim};
    margin-right: 8px;
    width: 0; height: 0;
}}
QComboBox QAbstractItemView {{
    background-color: {surface_hi};
    border: 1px solid {border};
    selection-background-color: {accent};
    selection-color: #FFFFFF;
    padding: 4px;
    outline: none;
}}

QCheckBox {{
    spacing: 8px;
    color: {text};
}}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {border};
    border-radius: 4px;
    background: {surface_hi};
}}
QCheckBox::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked {{
    background: {accent};
    border-color: {accent};
    image: none;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {border};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 2px;
}}
QSlider::add-page:horizontal {{
    background: {border};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px;
    margin: -6px 0;
    background: {text};
    border-radius: 7px;
    border: 2px solid {accent};
}}
QSlider::handle:horizontal:hover {{
    background: {accent_hi};
}}

/* ---------- project bar ---------- */

QFrame#ProjectBar {{
    background-color: {surface};
    border-bottom: 1px solid {border};
}}

/* ---------- log drawer ---------- */

QFrame#LogDrawer {{
    background-color: {surface};
    border-top: 1px solid {border};
}}
QFrame#LogHeader {{
    background-color: {surface};
    border-bottom: 1px solid {border};
}}
QPushButton#LogToggle {{
    background: transparent;
    border: none;
    color: {text_dim};
    font-size: 14px;
}}
QPushButton#LogToggle:hover {{ color: {text}; }}

QPushButton[role="ghost"] {{
    background: transparent;
    border: 1px solid transparent;
    color: {text_dim};
    padding: 4px 10px;
    border-radius: 4px;
}}
QPushButton[role="ghost"]:hover {{
    color: {text};
    border-color: {border};
}}

QLabel#LogDot {{
    font-size: 13px;
    padding: 0 2px;
}}
QLabel#LogDot[state="idle"]    {{ color: {text_dim}; }}
QLabel#LogDot[state="running"] {{ color: {accent}; }}
QLabel#LogDot[state="ok"]      {{ color: {success}; }}
QLabel#LogDot[state="err"]     {{ color: {error}; }}

QLabel[role="mono_dim"] {{
    font-family: "{mono}";
    color: {text_dim};
    font-size: 12px;
}}

QPlainTextEdit#LogArea {{
    border: none;
    border-radius: 0;
    background-color: {canvas};
    font-family: "{mono}";
    font-size: 12px;
    padding: 10px 14px;
}}

/* ---------- file path readout ---------- */

QLineEdit[role="readout"] {{
    background-color: {canvas};
    border: 1px dashed {border};
    color: {text_dim};
}}
"""
