"""
Theme — quiet, professional, minimal. A near-white canvas with a single
indigo accent, restrained chroma, generous spacing, and soft surfaces.
All colors live here so retheming is a single-file operation.
"""
from __future__ import annotations

from PySide6.QtGui import QPalette, QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


COLORS = {
    "canvas":     "#F7F7F5",
    "surface":    "#FFFFFF",
    "surface_hi": "#F1F1EE",
    "border":     "#E5E5E0",
    "border_hi":  "#D4D4CE",
    "text":       "#14141A",
    "text_mid":   "#4B4B55",
    "text_dim":   "#8A8A94",
    "accent":     "#4F46E5",
    "accent_hi":  "#6C63F0",
    "accent_soft":"#EEF0FF",
    "success":    "#0A7F4F",
    "success_bg": "#E7F5EE",
    "warning":    "#8A5A00",
    "warning_bg": "#FBF0DA",
    "error":      "#B42318",
    "error_bg":   "#FBE9E7",
}


def _pick_font(candidates: list[str], fallback: str) -> str:
    families = set(QFontDatabase.families())
    for name in candidates:
        if name in families:
            return name
    return fallback


def apply_theme(app: QApplication) -> None:
    ui = _pick_font(
        ["Inter", "SF Pro Text", "Segoe UI", "Ubuntu", "Noto Sans"],
        app.font().family(),
    )
    mono = _pick_font(
        ["JetBrains Mono", "SF Mono", "Menlo", "Ubuntu Mono", "DejaVu Sans Mono"],
        "monospace",
    )

    app.setFont(QFont(ui, 10))

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(COLORS["canvas"]))
    pal.setColor(QPalette.Base,            QColor(COLORS["surface"]))
    pal.setColor(QPalette.AlternateBase,   QColor(COLORS["surface_hi"]))
    pal.setColor(QPalette.Text,            QColor(COLORS["text"]))
    pal.setColor(QPalette.WindowText,      QColor(COLORS["text"]))
    pal.setColor(QPalette.Button,          QColor(COLORS["surface"]))
    pal.setColor(QPalette.ButtonText,      QColor(COLORS["text"]))
    pal.setColor(QPalette.Highlight,       QColor(COLORS["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.ToolTipBase,     QColor(COLORS["surface"]))
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

/* sidebar */
QFrame#Sidebar {{
    background-color: {surface};
    border-right: 1px solid {border};
}}
QLabel#AppTitle {{
    font-size: 15px;
    font-weight: 600;
    padding: 20px 20px 2px 20px;
    color: {text};
    letter-spacing: -0.2px;
}}
QLabel#AppTitleSub {{
    font-size: 10px;
    padding: 0 20px 18px 20px;
    color: {text_dim};
    letter-spacing: 1.2px;
}}
QPushButton[role="nav"] {{
    text-align: left;
    padding: 9px 16px;
    margin: 1px 10px;
    border: none;
    background: transparent;
    color: {text_mid};
    border-radius: 6px;
    font-weight: 500;
}}
QPushButton[role="nav"]:hover {{
    background-color: {surface_hi};
    color: {text};
}}
QPushButton[role="nav"][active="true"] {{
    color: {accent};
    background-color: {accent_soft};
    font-weight: 600;
}}

/* page */
QLabel#PageTitle {{
    font-size: 22px;
    font-weight: 600;
    color: {text};
    letter-spacing: -0.3px;
}}
QLabel[role="caption"] {{
    color: {text_dim};
    font-size: 12px;
}}

/* cards */
QFrame[role="card"] {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 10px;
}}
QFrame[role="subcard"] {{
    background-color: {canvas};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel[role="cardTitle"] {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.4px;
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

/* pills */
QLabel[pill="success"] {{
    background-color: {success_bg};
    color: {success};
    border: 1px solid {success_bg};
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 11px;
}}
QLabel[pill="warning"] {{
    background-color: {warning_bg};
    color: {warning};
    border: 1px solid {warning_bg};
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 11px;
}}
QLabel[pill="error"] {{
    background-color: {error_bg};
    color: {error};
    border: 1px solid {error_bg};
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 11px;
}}
QLabel[pill="neutral"] {{
    background-color: {surface_hi};
    color: {text_mid};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 3px 10px;
    font-size: 11px;
}}

/* buttons */
QPushButton[role="primary"] {{
    background-color: {accent};
    color: #FFFFFF;
    border: 1px solid {accent};
    border-radius: 7px;
    padding: 8px 18px;
    font-weight: 600;
}}
QPushButton[role="primary"]:hover {{ background-color: {accent_hi}; border-color: {accent_hi}; }}
QPushButton[role="primary"]:disabled {{
    background-color: {border};
    color: {text_dim};
    border-color: {border};
}}
QPushButton[role="secondary"] {{
    background-color: {surface};
    color: {text};
    border: 1px solid {border_hi};
    border-radius: 7px;
    padding: 8px 16px;
    font-weight: 500;
}}
QPushButton[role="secondary"]:hover {{
    border-color: {accent};
    color: {accent};
}}
QPushButton[role="secondary"]:disabled {{
    color: {text_dim};
    border-color: {border};
}}
QPushButton[role="ghost"] {{
    background: transparent;
    border: 1px solid transparent;
    color: {text_mid};
    padding: 6px 12px;
    border-radius: 6px;
}}
QPushButton[role="ghost"]:hover {{
    color: {text};
    background-color: {surface_hi};
}}

/* help badge */
QPushButton[role="help"] {{
    background-color: transparent;
    color: {text_dim};
    border: 1px solid {border_hi};
    border-radius: 8px;
    padding: 0;
    font-size: 10px;
    font-weight: 700;
    min-width: 16px;
    max-width: 16px;
    min-height: 16px;
    max-height: 16px;
}}
QPushButton[role="help"]:hover {{
    background-color: {accent_soft};
    color: {accent};
    border-color: {accent};
}}

/* inputs */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {surface};
    color: {text};
    border: 1px solid {border_hi};
    border-radius: 7px;
    padding: 7px 10px;
    selection-background-color: {accent_soft};
    selection-color: {accent};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {accent};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {text_dim};
    background-color: {surface_hi};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {text_mid};
    margin-right: 8px;
    width: 0; height: 0;
}}
QComboBox QAbstractItemView {{
    background-color: {surface};
    border: 1px solid {border_hi};
    selection-background-color: {accent_soft};
    selection-color: {accent};
    padding: 4px;
    outline: none;
}}

/* checkbox */
QCheckBox {{ spacing: 8px; color: {text}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {border_hi};
    border-radius: 4px;
    background: {surface};
}}
QCheckBox::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked {{
    background: {accent};
    border-color: {accent};
    image: none;
}}

/* slider */
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
    background: {surface};
    border-radius: 7px;
    border: 2px solid {accent};
}}
QSlider::handle:horizontal:hover {{ background: {accent_soft}; }}

/* text areas */
QPlainTextEdit, QTextEdit {{
    background-color: {surface};
    border: 1px solid {border_hi};
    border-radius: 7px;
    font-family: "{mono}";
    color: {text};
    padding: 8px;
}}

/* scrollbars */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {border_hi};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {text_dim}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    height: 0; background: transparent;
}}

/* tooltips */
QToolTip {{
    background-color: {text};
    color: #FFFFFF;
    border: none;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
}}

/* project bar */
QFrame#ProjectBar {{
    background-color: {surface};
    border-bottom: 1px solid {border};
}}

/* log drawer */
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
    color: {text};
}}

/* readout */
QLineEdit[role="readout"] {{
    background-color: {canvas};
    border: 1px solid {border};
    color: {text_mid};
    font-family: "{mono}";
    font-size: 12px;
}}

/* help popup */
QFrame#HelpPopup {{
    background-color: {surface};
    border: 1px solid {border_hi};
    border-radius: 10px;
}}
QLabel#HelpPopupTitle {{
    font-size: 13px;
    font-weight: 700;
    color: {text};
}}
QLabel#HelpPopupBody {{
    font-size: 12px;
    color: {text_mid};
}}
"""
