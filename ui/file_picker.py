"""
ui/file_picker.py
─────────────────
Reusable file-selection dialog for Sentinel.

Any feature that surfaces a list of Path objects can call:

    result = FilePickerDialog.ask(parent_widget, paths, title="…")
    if result:
        selected_paths, action = result
        # action is one of: "open", "copy_path", "show_in_explorer",
        #                    "move_to", "delete"

The dialog is self-contained — it handles its own styling and
needs no external state beyond the parent widget and the path list.
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QLineEdit, QFrame,
    QAbstractItemView, QHeaderView, QMessageBox, QFileDialog,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QFont, QIcon

# ── Palette (matches window.py) ───────────────────────────────────────
_BG_DEEP   = "#090b10"
_BG_PANEL  = "#0d1117"
_BG_CARD   = "#111827"
_BG_INPUT  = "#161e2e"
_ACCENT    = "#00d4ff"
_ACCENT2   = "#0080ff"
_SUCCESS   = "#10b981"
_ERROR     = "#ef4444"
_WARNING   = "#f59e0b"
_TEXT_PRI  = "#e2e8f0"
_TEXT_SEC  = "#94a3b8"
_TEXT_DIM  = "#475569"
_BORDER    = "#1e293b"

# ── File-type emoji ───────────────────────────────────────────────────
_EXT_ICON: dict = {
    # Documents
    "pdf": "📄", "doc": "📝", "docx": "📝", "txt": "📃",
    "md": "📃", "rtf": "📄", "odt": "📄",
    # Spreadsheets / data
    "xlsx": "📊", "xls": "📊", "csv": "📊", "json": "📋",
    "xml": "📋", "yaml": "📋", "yml": "📋",
    # Presentations
    "pptx": "📊", "ppt": "📊",
    # Images
    "png": "🖼", "jpg": "🖼", "jpeg": "🖼", "gif": "🖼",
    "bmp": "🖼", "svg": "🖼", "webp": "🖼", "ico": "🖼",
    # Audio / video
    "mp3": "🎵", "wav": "🎵", "flac": "🎵", "aac": "🎵",
    "mp4": "🎬", "mkv": "🎬", "avi": "🎬", "mov": "🎬",
    # Code
    "py": "🐍", "js": "📜", "ts": "📜", "html": "🌐",
    "css": "🎨", "cpp": "⚙", "c": "⚙", "cs": "⚙",
    "java": "☕", "rs": "🦀", "go": "🔵",
    # Archives
    "zip": "🗜", "rar": "🗜", "7z": "🗜", "tar": "🗜",
    "gz": "🗜",
    # Executables
    "exe": "⚙", "msi": "⚙", "sh": "⚙", "bat": "⚙",
}

def _ext_icon(path: Path) -> str:
    return _EXT_ICON.get(path.suffix.lstrip(".").lower(), "📄")

def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if n == int(n) else f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"

def _short_location(path: Path) -> str:
    """Return a compact parent path, relative to home if possible."""
    try:
        rel = path.parent.relative_to(Path.home())
        parts = rel.parts
        if not parts:
            return "~"
        if len(parts) <= 2:
            return "~/" + "/".join(parts)
        return f"~/{parts[0]}/…/{parts[-1]}"
    except ValueError:
        p = str(path.parent)
        return p[:40] + "…" if len(p) > 40 else p


# ─────────────────────────────────────────────
# Main dialog
# ─────────────────────────────────────────────

class FilePickerDialog(QDialog):
    """
    Modal file-selection dialog.

    Usage
    -----
    result = FilePickerDialog.ask(parent, paths)
    if result:
        selected, action = result
    """

    ACT_OPEN     = "open"
    ACT_COPY     = "copy_path"
    ACT_EXPLORER = "show_in_explorer"
    ACT_MOVE     = "move_to"
    ACT_DELETE   = "delete"

    def __init__(
        self,
        parent,
        paths: List[Path],
        title: str = "File Search Results",
    ):
        super().__init__(parent)
        self._all_paths = paths
        self._action    = ""
        self._build(title)
        self._apply_style()

    # ── Public API ────────────────────────────────────────────────────

    @staticmethod
    def ask(
        parent,
        paths: List[Path],
        title: str = "File Search Results",
    ) -> Optional[Tuple[List[Path], str]]:
        """
        Show the dialog and block until the user acts or dismisses it.

        Returns ``(selected_paths, action_id)`` or ``None`` if cancelled.
        """
        if not paths:
            return None
        dlg = FilePickerDialog(parent, paths, title)
        if dlg.exec_() == QDialog.DialogCode.Accepted:
            return dlg.selected_paths(), dlg._action
        return None

    def selected_paths(self) -> List[Path]:
        checked = []
        root = self._tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if item.checkState(0) == Qt.CheckState.Checked:
                p = item.data(0, Qt.ItemDataRole.UserRole)
                if p:
                    checked.append(Path(p))
        return checked

    # ── Build UI ──────────────────────────────────────────────────────

    def _build(self, title: str):
        self.setWindowTitle("Sentinel — File Results")
        self.setMinimumSize(800, 500)
        self.resize(920, 560)
        self.setModal(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        # ── Header ───────────────────────────────────────────────────
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(10)
        dot = QLabel("●")
        dot.setObjectName("PickerDot")
        dot.setFixedWidth(14)
        hdr_row.addWidget(dot)

        n = len(self._all_paths)
        hdr = QLabel(f"Found {n} file{'s' if n != 1 else ''} — select and choose an action")
        hdr.setObjectName("PickerHeader")
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()
        root.addLayout(hdr_row)

        # ── Filter bar ───────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)

        self._filter = QLineEdit()
        self._filter.setObjectName("PickerFilter")
        self._filter.setPlaceholderText("Filter by filename…")
        self._filter.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter, 1)

        btn_all = QPushButton("Select All")
        btn_all.setObjectName("PickerSelBtn")
        btn_all.setFixedHeight(32)
        btn_all.clicked.connect(lambda: self._set_all(True))
        filter_row.addWidget(btn_all)

        btn_none = QPushButton("None")
        btn_none.setObjectName("PickerSelBtn")
        btn_none.setFixedHeight(32)
        btn_none.clicked.connect(lambda: self._set_all(False))
        filter_row.addWidget(btn_none)

        root.addLayout(filter_row)

        # ── File tree ─────────────────────────────────────────────────
        self._tree = QTreeWidget()
        self._tree.setObjectName("PickerTree")
        self._tree.setColumnCount(5)
        self._tree.setHeaderLabels(["", "Filename", "Size", "Location", "Modified"])
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.setSortingEnabled(True)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        hdr_obj = self._tree.header()
        hdr_obj.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hdr_obj.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr_obj.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hdr_obj.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hdr_obj.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self._tree.setColumnWidth(0, 28)
        self._tree.setColumnWidth(2, 80)
        self._tree.setColumnWidth(4, 100)

        self._tree.itemChanged.connect(self._on_item_changed)
        root.addWidget(self._tree, 1)

        # ── Status row ────────────────────────────────────────────────
        status_row = QHBoxLayout()
        self._count_label = QLabel("0 selected")
        self._count_label.setObjectName("PickerCount")
        status_row.addWidget(self._count_label)
        status_row.addStretch()
        root.addLayout(status_row)

        # ── Separator ─────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("PickerSep")
        root.addWidget(sep)

        # ── Action buttons ────────────────────────────────────────────
        act_row = QHBoxLayout()
        act_row.setSpacing(8)

        def _btn(label: str, obj: str, action: str, danger: bool = False) -> QPushButton:
            b = QPushButton(label)
            b.setObjectName("ActDanger" if danger else "ActBtn")
            b.setFixedHeight(38)
            b.clicked.connect(lambda: self._accept(action))
            return b

        act_row.addWidget(_btn("Open",           "ActBtn",     self.ACT_OPEN))
        act_row.addWidget(_btn("Copy Path",       "ActBtn",     self.ACT_COPY))
        act_row.addWidget(_btn("Show in Folder",  "ActBtn",     self.ACT_EXPLORER))
        act_row.addWidget(_btn("Move to…",        "ActBtn",     self.ACT_MOVE))
        act_row.addStretch()
        act_row.addWidget(_btn("Delete",          "ActDanger",  self.ACT_DELETE, danger=True))

        root.addLayout(act_row)

        # Populate after all widgets exist (count_label must be constructed first)
        self._populate()

    # ── Tree population ───────────────────────────────────────────────

    def _populate(self, filter_text: str = ""):
        self._tree.blockSignals(True)
        self._tree.clear()

        q = filter_text.strip().lower()
        for path in self._all_paths:
            if q and q not in path.name.lower():
                continue

            item = QTreeWidgetItem()
            item.setCheckState(0, Qt.CheckState.Unchecked)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
            )

            # Icon + filename
            item.setText(1, f"{_ext_icon(path)}  {path.name}")

            # Size / modified
            try:
                st = path.stat()
                item.setText(2, _fmt_size(st.st_size))
                item.setText(4, datetime.fromtimestamp(st.st_mtime).strftime("%b %d, %Y"))
            except OSError:
                item.setText(2, "—")
                item.setText(4, "—")

            item.setText(3, _short_location(path))
            item.setData(0, Qt.ItemDataRole.UserRole, str(path))

            # Right-align the size column
            item.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            self._tree.addTopLevelItem(item)

        self._tree.blockSignals(False)
        self._update_count()

    # ── Slot helpers ──────────────────────────────────────────────────

    def _apply_filter(self, text: str):
        self._populate(text)

    def _set_all(self, checked: bool):
        self._tree.blockSignals(True)
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        root  = self._tree.invisibleRootItem()
        for i in range(root.childCount()):
            root.child(i).setCheckState(0, state)
        self._tree.blockSignals(False)
        self._update_count()

    def _on_item_changed(self, item: QTreeWidgetItem, col: int):
        if col == 0:
            self._update_count()

    def _update_count(self):
        n = len(self.selected_paths())
        self._count_label.setText(f"{n} file{'s' if n != 1 else ''} selected")

    def _accept(self, action: str):
        sel = self.selected_paths()
        if not sel:
            self._count_label.setText("Select at least one file first.")
            self._count_label.setStyleSheet(f"color: {_WARNING}; font-size: 12px;")
            return
        self._action = action
        self.accept()

    # ── Stylesheet ────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet(f"""
        QDialog {{
            background: {_BG_DEEP};
            color: {_TEXT_PRI};
            font-family: 'Segoe UI', 'Inter', sans-serif;
        }}

        #PickerDot {{
            font-size: 10px;
            color: {_ACCENT};
            background: transparent;
        }}

        #PickerHeader {{
            font-size: 15px;
            font-weight: 600;
            color: {_TEXT_PRI};
            background: transparent;
        }}

        #PickerFilter {{
            background: {_BG_INPUT};
            border: 1px solid {_BORDER};
            border-radius: 8px;
            color: {_TEXT_PRI};
            font-size: 13px;
            padding: 6px 14px;
            height: 32px;
        }}
        #PickerFilter:focus {{
            border: 1px solid rgba(0, 212, 255, 0.6);
        }}

        #PickerSelBtn {{
            background: {_BG_CARD};
            border: 1px solid {_BORDER};
            border-radius: 6px;
            color: {_TEXT_SEC};
            font-size: 12px;
            padding: 0 12px;
        }}
        #PickerSelBtn:hover {{
            background: {_BORDER};
            color: {_TEXT_PRI};
        }}

        #PickerTree {{
            background: {_BG_PANEL};
            alternate-background-color: {_BG_CARD};
            border: 1px solid {_BORDER};
            border-radius: 8px;
            color: {_TEXT_PRI};
            font-size: 13px;
            outline: none;
        }}
        #PickerTree::item {{
            padding: 5px 4px;
            border: none;
        }}
        #PickerTree::item:selected {{
            background: rgba(0, 212, 255, 0.10);
            color: {_TEXT_PRI};
        }}
        #PickerTree::item:hover {{
            background: rgba(255, 255, 255, 0.04);
        }}

        QHeaderView::section {{
            background: {_BG_CARD};
            color: {_TEXT_DIM};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
            padding: 6px 8px;
            border: none;
            border-right: 1px solid {_BORDER};
            border-bottom: 1px solid {_BORDER};
        }}

        QScrollBar:vertical {{
            background: {_BG_PANEL};
            width: 6px;
            border-radius: 3px;
        }}
        QScrollBar::handle:vertical {{
            background: {_BORDER};
            border-radius: 3px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {_TEXT_DIM};
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{ height: 0; }}

        #PickerCount {{
            font-size: 12px;
            color: {_TEXT_DIM};
            background: transparent;
        }}

        #PickerSep {{
            background: {_BORDER};
            max-height: 1px;
        }}

        #ActBtn {{
            background: {_BG_CARD};
            border: 1px solid {_BORDER};
            border-radius: 8px;
            color: {_TEXT_SEC};
            font-size: 13px;
            padding: 0 18px;
        }}
        #ActBtn:hover {{
            background: rgba(0, 212, 255, 0.10);
            border: 1px solid rgba(0, 212, 255, 0.35);
            color: {_ACCENT};
        }}
        #ActBtn:pressed {{
            background: rgba(0, 212, 255, 0.18);
        }}

        #ActDanger {{
            background: rgba(239, 68, 68, 0.08);
            border: 1px solid rgba(239, 68, 68, 0.25);
            border-radius: 8px;
            color: {_ERROR};
            font-size: 13px;
            padding: 0 18px;
        }}
        #ActDanger:hover {{
            background: rgba(239, 68, 68, 0.18);
            border: 1px solid rgba(239, 68, 68, 0.55);
        }}
        """)


# ─────────────────────────────────────────────
# Action executor  (called from window.py)
# ─────────────────────────────────────────────

class FileActionExecutor:
    """
    Executes the action chosen in FilePickerDialog.
    Designed to be called from the main (UI) thread.

    Returns (success: bool, message: str).
    """

    @staticmethod
    def run(
        action: str,
        paths:  List[Path],
        parent_widget=None,
    ) -> Tuple[bool, str]:
        if not paths:
            return False, "No files selected."

        if action == FilePickerDialog.ACT_OPEN:
            return FileActionExecutor._open(paths)

        if action == FilePickerDialog.ACT_COPY:
            return FileActionExecutor._copy_paths(paths, parent_widget)

        if action == FilePickerDialog.ACT_EXPLORER:
            return FileActionExecutor._show_in_explorer(paths[0])

        if action == FilePickerDialog.ACT_MOVE:
            return FileActionExecutor._move(paths, parent_widget)

        if action == FilePickerDialog.ACT_DELETE:
            return FileActionExecutor._delete(paths, parent_widget)

        return False, f"Unknown action: {action}"

    # ── Individual handlers ───────────────────────────────────────────

    @staticmethod
    def _open(paths: List[Path]) -> Tuple[bool, str]:
        errors = []
        for p in paths:
            try:
                if sys.platform == "win32":
                    os.startfile(str(p))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(p)])
                else:
                    subprocess.Popen(["xdg-open", str(p)])
            except Exception as e:
                errors.append(str(e))
        if errors:
            return False, f"Could not open {len(errors)} file(s): {errors[0]}"
        n = len(paths)
        return True, f"Opened {n} file{'s' if n != 1 else ''}."

    @staticmethod
    def _copy_paths(paths: List[Path], parent) -> Tuple[bool, str]:
        from PyQt5.QtWidgets import QApplication
        text = "\n".join(str(p) for p in paths)
        QApplication.clipboard().setText(text)
        n = len(paths)
        return True, f"Copied {n} path{'s' if n != 1 else ''} to clipboard."

    @staticmethod
    def _show_in_explorer(path: Path) -> Tuple[bool, str]:
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
            return True, f"Opened folder for '{path.name}'."
        except Exception as e:
            return False, str(e)

    @staticmethod
    def _move(paths: List[Path], parent) -> Tuple[bool, str]:
        dest_str = QFileDialog.getExistingDirectory(parent, "Move files to…")
        if not dest_str:
            return False, "Move cancelled."
        dest = Path(dest_str)
        moved, errors = 0, []
        for p in paths:
            target = dest / p.name
            if target.exists():
                stem, suf = p.stem, p.suffix
                import time as _t
                target = dest / f"{stem}_{int(_t.time())}{suf}"
            try:
                shutil.move(str(p), str(target))
                moved += 1
            except Exception as e:
                errors.append(f"{p.name}: {e}")
        msg = f"Moved {moved} file{'s' if moved != 1 else ''} to '{dest.name}'."
        if errors:
            msg += f"\n{len(errors)} error(s): {errors[0]}"
        return moved > 0, msg

    @staticmethod
    def _delete(paths: List[Path], parent) -> Tuple[bool, str]:
        preview = "\n".join(f"  • {p.name}" for p in paths[:8])
        if len(paths) > 8:
            preview += f"\n  … and {len(paths) - 8} more"
        reply = QMessageBox.question(
            parent,
            "Confirm Delete",
            f"Permanently delete {len(paths)} file{'s' if len(paths) != 1 else ''}?\n\n{preview}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False, "Delete cancelled."
        deleted, errors = 0, []
        for p in paths:
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                errors.append(f"{p.name}: {e}")
        msg = f"Deleted {deleted} file{'s' if deleted != 1 else ''}."
        if errors:
            msg += f"\n{len(errors)} could not be deleted."
        return deleted > 0, msg
