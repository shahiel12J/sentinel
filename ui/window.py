"""
Sentinel UI — Main Window

Jarvis-style dark interface with:
  • Animated status orb
  • Animated 3-D particle-sphere background
  • Natural-language command input
  • Real-time action log (step-by-step)
  • Collapsible memory / recent sidebar
  • System status bar
"""

import math
from datetime import datetime
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextEdit, QLabel, QSplitter, QFrame,
    QScrollArea, QSizePolicy, QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal,
)
from PyQt5.QtGui import (
    QColor, QPainter, QBrush, QRadialGradient,
)


# ─────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────

BG_DEEP    = "#090b10"
BG_PANEL   = "#0d1117"
BG_CARD    = "#111827"
BG_INPUT   = "#161e2e"
ACCENT_1   = "#00d4ff"
ACCENT_2   = "#0080ff"
ACCENT_3   = "#7c3aed"
SUCCESS    = "#10b981"
WARNING    = "#f59e0b"
ERROR      = "#ef4444"
TEXT_PRI   = "#e2e8f0"
TEXT_SEC   = "#94a3b8"
TEXT_DIM   = "#475569"
BORDER     = "#1e293b"
GLOW_CYAN  = "#00d4ff40"


# ─────────────────────────────────────────────
# Shared visual state constants
# ─────────────────────────────────────────────

STATE_IDLE      = "idle"
STATE_THINKING  = "thinking"
STATE_EXECUTING = "executing"
STATE_SUCCESS   = "success"
STATE_ERROR     = "error"


# ─────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────

class AgentWorker(QThread):
    """Runs Sentinel's pipeline off the main thread."""
    step_start  = pyqtSignal(int, str)
    step_done   = pyqtSignal(int, str, str)
    plan_done   = pyqtSignal(bool, str)
    output_text = pyqtSignal(str)
    preamble    = pyqtSignal(str)

    def __init__(self, agent, command: str):
        super().__init__()
        self.agent   = agent
        self.command = command

    def run(self):
        self.agent.process_command(
            command       = self.command,
            on_preamble   = self.preamble.emit,
            on_step_start = self.step_start.emit,
            on_step_done  = self.step_done.emit,
            on_plan_done  = self.plan_done.emit,
            on_output     = self.output_text.emit,
        )


# ─────────────────────────────────────────────
# Pulsing orb widget
# ─────────────────────────────────────────────

class OrbWidget(QWidget):
    """Animated cyan orb indicating Sentinel's state."""

    IDLE      = STATE_IDLE
    THINKING  = STATE_THINKING
    EXECUTING = STATE_EXECUTING
    SUCCESS   = STATE_SUCCESS
    ERROR     = STATE_ERROR

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 80)
        self._state   = self.IDLE
        self._phase   = 0.0
        self._timer   = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def set_state(self, state: str):
        self._state = state

    def _tick(self):
        self._phase = (self._phase + 0.04) % (2 * math.pi)
        self.update()

    def paintEvent(self, a0):  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2

        pulse = 0.5 + 0.5 * math.sin(self._phase)

        if self._state == self.IDLE:
            core_color  = QColor(ACCENT_1)
            glow_alpha  = int(40 + 30 * pulse)
            core_radius = 14
        elif self._state == self.THINKING:
            core_color  = QColor(ACCENT_3)
            glow_alpha  = int(60 + 40 * pulse)
            core_radius = int(14 + 3 * pulse)
        elif self._state == self.EXECUTING:
            core_color  = QColor(ACCENT_2)
            glow_alpha  = int(80 + 60 * pulse)
            core_radius = int(16 + 4 * pulse)
        elif self._state == self.SUCCESS:
            core_color  = QColor(SUCCESS)
            glow_alpha  = int(60 + 30 * pulse)
            core_radius = 14
        else:
            core_color  = QColor(ERROR)
            glow_alpha  = int(80 + 40 * pulse)
            core_radius = 14

        for r, a in [(38, glow_alpha // 4), (28, glow_alpha // 2), (20, glow_alpha)]:
            glow = QColor(core_color)
            glow.setAlpha(a)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(int(cx - r), int(cy - r), r * 2, r * 2)

        grad = QRadialGradient(cx - 3, cy - 3, core_radius * 1.5)
        bright = QColor(core_color)
        bright.setAlpha(255)
        dim = QColor(core_color)
        dim.setAlpha(180)
        grad.setColorAt(0, QColor("#ffffff"))
        grad.setColorAt(0.3, bright)
        grad.setColorAt(1, dim)
        p.setBrush(QBrush(grad))
        p.drawEllipse(int(cx - core_radius), int(cy - core_radius),
                      core_radius * 2, core_radius * 2)
        p.end()


# ─────────────────────────────────────────────
# Step card widget
# ─────────────────────────────────────────────

class StepCard(QFrame):
    """One step in the action log."""

    STATUS_COLORS = {
        "pending":  TEXT_DIM,
        "running":  ACCENT_1,
        "done":     SUCCESS,
        "failed":   ERROR,
        "skipped":  TEXT_DIM,
    }

    def __init__(self, step_num: int, description: str, parent=None):
        super().__init__(parent)
        self.step_num = step_num
        self.setObjectName("StepCard")
        self.setStyleSheet(f"""
            #StepCard {{
                background: {BG_CARD};
                border: 1px solid {BORDER};
                border-left: 3px solid {TEXT_DIM};
                border-radius: 6px;
                padding: 0px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        self.badge = QLabel(f"{step_num + 1}")
        self.badge.setFixedSize(22, 22)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setStyleSheet(f"""
            background: {BORDER}; color: {TEXT_DIM};
            border-radius: 11px; font-size: 10px; font-weight: bold;
        """)

        self.desc_label = QLabel(description)
        self.desc_label.setStyleSheet(f"color: {TEXT_SEC}; font-size: 12px;")
        self.desc_label.setWordWrap(True)

        self.status_label = QLabel("○")
        self.status_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 14px; min-width: 18px;")
        self.status_label.setAlignment(Qt.AlignmentFlag(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))

        self.msg_label = QLabel("")
        self.msg_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self.msg_label.setWordWrap(True)

        content = QVBoxLayout()
        content.setSpacing(2)
        content.addWidget(self.desc_label)
        content.addWidget(self.msg_label)

        layout.addWidget(self.badge)
        layout.addLayout(content, 1)
        layout.addWidget(self.status_label)

    def set_running(self):
        self._set_border(ACCENT_1)
        self.badge.setStyleSheet(f"background:{ACCENT_1}; color:#000; border-radius:11px; font-size:10px; font-weight:bold;")
        self.status_label.setText("◉")
        self.status_label.setStyleSheet(f"color:{ACCENT_1}; font-size:14px; min-width:18px;")

    def set_done(self, message: str = ""):
        self._set_border(SUCCESS)
        self.badge.setStyleSheet(f"background:{SUCCESS}; color:#000; border-radius:11px; font-size:10px; font-weight:bold;")
        self.status_label.setText("●")
        self.status_label.setStyleSheet(f"color:{SUCCESS}; font-size:14px; min-width:18px;")
        if message:
            short = message[:120] + ("…" if len(message) > 120 else "")
            self.msg_label.setText(short)
            self.msg_label.setStyleSheet(f"color:{TEXT_SEC}; font-size:11px;")

    def set_failed(self, message: str = ""):
        self._set_border(ERROR)
        self.badge.setStyleSheet(f"background:{ERROR}; color:#fff; border-radius:11px; font-size:10px; font-weight:bold;")
        self.status_label.setText("✗")
        self.status_label.setStyleSheet(f"color:{ERROR}; font-size:14px; min-width:18px;")
        if message:
            self.msg_label.setText(message[:120])
            self.msg_label.setStyleSheet(f"color:{ERROR}; font-size:11px;")

    def _set_border(self, color: str):
        self.setStyleSheet(f"""
            #StepCard {{
                background:{BG_CARD}; border:1px solid {BORDER};
                border-left:3px solid {color}; border-radius:6px;
            }}
        """)


# ─────────────────────────────────────────────
# Particle sphere background
# ─────────────────────────────────────────────

class ParticleSphereWidget(QWidget):
    """
    Animated 3-D particle sphere rendered with QPainter.
    Uses a Fibonacci lattice for uniform dot distribution and
    dual-frequency sine-wave distortion for an organic look.
    """

    _COLORS = {
        STATE_IDLE:      (0,   212, 255),
        STATE_THINKING:  (124,  58, 237),
        STATE_EXECUTING: (0,   128, 255),
        STATE_SUCCESS:   (16,  185, 129),
        STATE_ERROR:     (239,  68,  68),
    }
    _WAVE_AMP = {
        STATE_IDLE:      0.04,
        STATE_THINKING:  0.13,
        STATE_EXECUTING: 0.20,
        STATE_SUCCESS:   0.06,
        STATE_ERROR:     0.15,
    }
    _SPEED = {
        STATE_IDLE:      (0.006, 0.012),
        STATE_THINKING:  (0.018, 0.038),
        STATE_EXECUTING: (0.026, 0.058),
        STATE_SUCCESS:   (0.008, 0.018),
        STATE_ERROR:     (0.022, 0.048),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._state    = STATE_IDLE
        self._phase    = 0.0
        self._rotation = 0.0

        # Pre-compute per-point constants (Fibonacci sphere, N=480)
        N = 480
        golden = (1.0 + math.sqrt(5.0)) / 2.0
        self._pts: list = []
        for i in range(N):
            theta = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (i + 0.5) / N)))
            phi   = 2.0 * math.pi * i / golden
            self._pts.append((theta, phi, math.sin(theta), math.cos(theta)))

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)          # ~30 fps

    def set_state(self, state: str):
        self._state = state

    def _tick(self):
        rot_s, phase_s = self._SPEED.get(self._state, (0.006, 0.012))
        self._rotation = (self._rotation + rot_s)  % (2.0 * math.pi)
        self._phase    = (self._phase    + phase_s) % (2.0 * math.pi)
        self.update()

    def paintEvent(self, _event):  # type: ignore[override]
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)

        cx     = w / 2.0
        cy     = h / 2.0
        base_r = min(w, h) * 0.40
        fov    = base_r * 5.0
        wave_a = self._WAVE_AMP.get(self._state, 0.04)
        dr, dg, db = self._COLORS.get(self._state, (0, 212, 255))
        rot    = self._rotation
        phase  = self._phase

        for theta, phi, sin_t, cos_t in self._pts:
            phi_r  = phi + rot
            sin_pr = math.sin(phi_r)
            cos_pr = math.cos(phi_r)

            wave = (wave_a       * math.sin(2.0 * theta + phase)        * math.cos(3.0 * phi_r + phase * 0.70)
                  + wave_a * 0.4 * math.sin(4.0 * theta - phase * 1.20) * math.cos(phi_r       + phase * 0.50))
            r = base_r * (1.0 + wave)

            x =  r * sin_t * sin_pr
            y =  r * cos_t
            z =  r * sin_t * cos_pr

            scale = fov / (fov + z + base_r)
            sx    = cx + x * scale
            sy    = cy - y * scale

            depth = max(0.0, min(1.0, (z + base_r * 1.1) / (base_r * 2.2)))
            alpha = int(12 + 210 * depth)
            dsz   = max(1, int(0.8 + 3.4 * depth))

            p.setBrush(QBrush(QColor(dr, dg, db, alpha)))
            half = dsz // 2
            p.drawEllipse(int(sx - half), int(sy - half), dsz, dsz)

        p.end()


# ─────────────────────────────────────────────
# Main-panel container (positions sphere overlay)
# ─────────────────────────────────────────────

class MainPanelWidget(QWidget):
    """
    Hosts the main panel layout AND an absolute-positioned
    ParticleSphereWidget that tracks the log scroll area geometry.
    """

    def __init__(self):
        super().__init__()
        self.sphere   = ParticleSphereWidget(self)
        self._log_ref: Optional[QWidget] = None

    def attach_sphere(self, log_scroll: QWidget):
        self._log_ref = log_scroll
        self.sphere.lower()
        QTimer.singleShot(0, self._sync_sphere)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        QTimer.singleShot(0, self._sync_sphere)

    def _sync_sphere(self):
        if self._log_ref is not None:
            self.sphere.setGeometry(self._log_ref.geometry())


# ─────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────

class SentinelWindow(QMainWindow):

    def __init__(self, agent):
        super().__init__()
        self.agent              = agent
        self._worker:           Optional[AgentWorker] = None
        self._step_cards        = []
        self._sidebar_last_size = 290

        self._build_ui()
        self._apply_stylesheet()

        clock_timer = QTimer(self)
        clock_timer.timeout.connect(self._update_clock)
        clock_timer.start(1000)
        self._update_clock()

        mem_timer = QTimer(self)
        mem_timer.timeout.connect(self._refresh_memory)
        mem_timer.start(5000)

    # ── Build UI ─────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("Sentinel — Desktop AI Agent")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 780)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_topbar())
        root.addWidget(self._build_body(), 1)
        root.addWidget(self._build_statusbar())

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(64)
        bar.setObjectName("TopBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(16)

        self.orb = OrbWidget()
        lay.addWidget(self.orb)

        # Thin vertical separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet(f"background: {BORDER}; margin: 14px 0px;")
        lay.addWidget(sep)

        title_block = QVBoxLayout()
        title_block.setSpacing(3)
        title = QLabel("SENTINEL")
        title.setObjectName("AppTitle")

        # Cyan glow behind the title text
        _glow = QGraphicsDropShadowEffect()
        _glow.setColor(QColor(ACCENT_1))
        _glow.setBlurRadius(22)
        _glow.setOffset(0, 0)
        title.setGraphicsEffect(_glow)

        # Subtitle as three small pill badges
        badge_row = QWidget()
        badge_row.setStyleSheet("background: transparent;")
        badge_lay = QHBoxLayout(badge_row)
        badge_lay.setContentsMargins(0, 0, 0, 0)
        badge_lay.setSpacing(6)
        for tag in ["Desktop AI Agent", "Local", "Always On"]:
            b = QLabel(tag)
            b.setObjectName("SubtitleBadge")
            badge_lay.addWidget(b)
        badge_lay.addStretch()

        title_block.addWidget(title)
        title_block.addWidget(badge_row)
        lay.addLayout(title_block)

        lay.addStretch()

        mode = getattr(self.agent.classifier, "mode", "rule-based")
        self.mode_badge = QLabel(f"⚡ {mode.upper()}")
        self.mode_badge.setObjectName("ModeBadge")
        lay.addWidget(self.mode_badge)

        self.clock_label = QLabel()
        self.clock_label.setObjectName("Clock")
        lay.addWidget(self.clock_label)

        return bar

    def _build_body(self) -> QWidget:
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(1)
        self._splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        # ── Left: main panel ─────────────────────────────────────────
        self.main_panel = MainPanelWidget()
        self.main_panel.setObjectName("MainPanel")
        main_lay = QVBoxLayout(self.main_panel)
        main_lay.setContentsMargins(20, 16, 20, 16)
        main_lay.setSpacing(12)

        response_frame = QFrame()
        response_frame.setObjectName("ResponseFrame")
        resp_inner = QHBoxLayout(response_frame)
        resp_inner.setContentsMargins(16, 10, 16, 10)
        resp_inner.setSpacing(10)

        resp_dot = QLabel("●")
        resp_dot.setObjectName("ResponseDot")
        resp_dot.setFixedWidth(12)
        resp_inner.addWidget(resp_dot)

        self.response_label = QLabel("Ready. Type a command below.")
        self.response_label.setObjectName("ResponseLabel")
        self.response_label.setWordWrap(True)
        self.response_label.setAlignment(Qt.AlignmentFlag(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter))
        resp_inner.addWidget(self.response_label, 1)
        main_lay.addWidget(response_frame)

        self.log_scroll = QScrollArea()
        self.log_scroll.setObjectName("LogScroll")
        self.log_scroll.setWidgetResizable(True)
        self.log_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.log_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.log_scroll.setMinimumHeight(200)
        self.log_scroll.setAutoFillBackground(False)
        vp = self.log_scroll.viewport()
        if vp is not None:
            vp.setAutoFillBackground(False)

        self.log_container = QWidget()
        self.log_container.setObjectName("LogContainer")
        self.log_container.setAutoFillBackground(False)
        self.log_layout   = QVBoxLayout(self.log_container)
        self.log_layout.setContentsMargins(0, 0, 0, 0)
        self.log_layout.setSpacing(6)
        self.log_layout.addStretch()
        self.log_scroll.setWidget(self.log_container)
        main_lay.addWidget(self.log_scroll, 1)

        self.output_box = QTextEdit()
        self.output_box.setObjectName("OutputBox")
        self.output_box.setReadOnly(True)
        self.output_box.setMaximumHeight(200)
        self.output_box.setVisible(False)
        main_lay.addWidget(self.output_box)

        input_row = QHBoxLayout()
        input_row.setSpacing(10)

        self.input_field = QLineEdit()
        self.input_field.setObjectName("CommandInput")
        self.input_field.setPlaceholderText("Type a command  (e.g.  open chrome,  find all PDFs,  shutdown in 30 minutes)")
        self.input_field.returnPressed.connect(self._on_send)

        self.send_btn = QPushButton("Execute")
        self.send_btn.setObjectName("SendButton")
        self.send_btn.setFixedHeight(42)
        self.send_btn.setMinimumWidth(110)
        self.send_btn.clicked.connect(self._on_send)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("ClearButton")
        self.clear_btn.setFixedHeight(42)
        self.clear_btn.setMinimumWidth(70)
        self.clear_btn.clicked.connect(self._clear_log)

        input_row.addWidget(self.input_field, 1)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(self.clear_btn)
        main_lay.addLayout(input_row)

        self.main_panel.attach_sphere(self.log_scroll)
        self._splitter.addWidget(self.main_panel)

        # ── Right: sidebar container (tab strip + content) ────────────
        sidebar_container = QWidget()
        sidebar_container.setObjectName("SidebarContainer")
        sidebar_container.setMinimumWidth(26)

        outer_lay = QHBoxLayout(sidebar_container)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(0)

        # Persistent collapse tab — always visible
        self._sidebar_tab = QPushButton("◀")
        self._sidebar_tab.setObjectName("SidebarTab")
        self._sidebar_tab.setFixedWidth(24)
        self._sidebar_tab.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        self._sidebar_tab.setToolTip("Toggle sidebar")
        self._sidebar_tab.clicked.connect(self._toggle_sidebar)
        outer_lay.addWidget(self._sidebar_tab)

        # Sidebar content panel
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setMinimumWidth(260)
        self.sidebar.setMaximumWidth(320)
        side_lay = QVBoxLayout(self.sidebar)
        side_lay.setContentsMargins(12, 16, 12, 16)
        side_lay.setSpacing(8)

        # Memory header + toggle
        mem_hdr = QHBoxLayout()
        mem_hdr.setContentsMargins(0, 0, 0, 0)
        mem_title = QLabel("🧠  MEMORY")
        mem_title.setObjectName("SectionTitle")
        self._mem_toggle = QPushButton("▼")
        self._mem_toggle.setObjectName("ToggleBtn")
        self._mem_toggle.setFixedSize(18, 18)
        self._mem_toggle.clicked.connect(self._toggle_memory)
        mem_hdr.addWidget(mem_title)
        mem_hdr.addStretch()
        mem_hdr.addWidget(self._mem_toggle)
        side_lay.addLayout(mem_hdr)

        self.memory_box = QTextEdit()
        self.memory_box.setObjectName("MemoryBox")
        self.memory_box.setReadOnly(True)
        self.memory_box.setPlaceholderText("No memories yet.\nTry: 'remember my IDE is Rider'")
        side_lay.addWidget(self.memory_box, 1)

        # Recent header + toggle
        hist_hdr = QHBoxLayout()
        hist_hdr.setContentsMargins(0, 0, 0, 0)
        hist_title = QLabel("🕑  RECENT")
        hist_title.setObjectName("SectionTitle")
        self._hist_toggle = QPushButton("▼")
        self._hist_toggle.setObjectName("ToggleBtn")
        self._hist_toggle.setFixedSize(18, 18)
        self._hist_toggle.clicked.connect(self._toggle_recent)
        hist_hdr.addWidget(hist_title)
        hist_hdr.addStretch()
        hist_hdr.addWidget(self._hist_toggle)
        side_lay.addLayout(hist_hdr)

        self.history_box = QTextEdit()
        self.history_box.setObjectName("HistoryBox")
        self.history_box.setReadOnly(True)
        self.history_box.setMaximumHeight(140)
        side_lay.addWidget(self.history_box)

        outer_lay.addWidget(self.sidebar, 1)
        self._splitter.addWidget(sidebar_container)
        self._splitter.setSizes([860, 300])

        return self._splitter

    def _build_statusbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setObjectName("StatusBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(24)

        self.status_label = QLabel("Sentinel is ready.")
        self.status_label.setObjectName("StatusLabel")
        lay.addWidget(self.status_label)

        lay.addStretch()

        gpu_lbl = QLabel("GPU: NVIDIA 3050  ·  Local AI")
        gpu_lbl.setObjectName("StatusInfo")
        lay.addWidget(gpu_lbl)

        return bar

    # ── Stylesheet ────────────────────────────────────────────────────

    def _apply_stylesheet(self):
        self.setStyleSheet(f"""
        QMainWindow, QWidget {{
            background: {BG_DEEP};
            color: {TEXT_PRI};
            font-family: 'Segoe UI', 'Inter', sans-serif;
        }}

        #TopBar {{
            background: {BG_PANEL};
            border-bottom: 1px solid {BORDER};
        }}

        #AppTitle {{
            font-size: 23px;
            font-weight: 800;
            letter-spacing: 6px;
            color: {ACCENT_1};
        }}

        #SubtitleBadge {{
            font-size: 9px;
            font-weight: 600;
            letter-spacing: 0.5px;
            color: {TEXT_DIM};
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1px 7px;
        }}

        #ModeBadge {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 10px;
            font-weight: bold;
            color: {ACCENT_1};
        }}

        #Clock {{
            font-size: 14px;
            color: {TEXT_SEC};
            min-width: 120px;
        }}

        #MainPanel {{
            background: {BG_DEEP};
        }}

        #ResponseFrame {{
            background: rgba(9, 11, 16, 0.55);
            border: 1px solid rgba(0, 212, 255, 0.14);
            border-left: 3px solid rgba(0, 212, 255, 0.55);
            border-radius: 8px;
        }}

        #ResponseDot {{
            font-size: 9px;
            color: {ACCENT_1};
            background: transparent;
        }}

        #ResponseLabel {{
            font-size: 14px;
            font-weight: 500;
            color: {TEXT_PRI};
            background: transparent;
            min-height: 22px;
        }}

        #LogScroll {{
            background: transparent;
            border: 1px solid rgba(30, 41, 59, 0.5);
            border-radius: 8px;
        }}

        QScrollArea > QWidget > QWidget {{
            background: transparent;
        }}

        #LogContainer {{
            background: transparent;
        }}

        QScrollBar:vertical {{
            background: {BG_PANEL};
            width: 6px;
            border-radius: 3px;
        }}
        QScrollBar::handle:vertical {{
            background: {BORDER};
            border-radius: 3px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {TEXT_DIM};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

        #OutputBox {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 8px;
            font-family: 'Consolas', 'Cascadia Code', monospace;
            font-size: 12px;
            color: {TEXT_PRI};
            padding: 8px;
        }}

        #CommandInput {{
            background: rgba(22, 30, 46, 0.85);
            border: 1px solid {BORDER};
            border-radius: 10px;
            color: {TEXT_PRI};
            font-size: 14px;
            padding: 11px 18px;
            height: 44px;
        }}
        #CommandInput:focus {{
            border: 1px solid rgba(0, 212, 255, 0.7);
            background: rgba(22, 30, 46, 0.95);
        }}
        #CommandInput::placeholder {{
            color: {TEXT_DIM};
        }}

        #SendButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {ACCENT_2}, stop:1 {ACCENT_1});
            border: none;
            border-radius: 8px;
            color: #000;
            font-weight: bold;
            font-size: 13px;
            letter-spacing: 1px;
        }}
        #SendButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {ACCENT_1}, stop:1 #00ffff);
        }}
        #SendButton:pressed {{
            background: {ACCENT_2};
        }}
        #SendButton:disabled {{
            background: {BORDER};
            color: {TEXT_DIM};
        }}

        #ClearButton {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 8px;
            color: {TEXT_SEC};
            font-size: 12px;
        }}
        #ClearButton:hover {{
            background: {BORDER};
            color: {TEXT_PRI};
        }}

        #SidebarContainer {{
            background: {BG_PANEL};
            border-left: 1px solid {BORDER};
        }}

        #SidebarTab {{
            background: {BG_PANEL};
            border: none;
            border-right: 1px solid {BORDER};
            color: {TEXT_DIM};
            font-size: 9px;
        }}
        #SidebarTab:hover {{
            background: {BG_CARD};
            color: {ACCENT_1};
        }}

        #Sidebar {{
            background: {BG_PANEL};
        }}

        #SectionTitle {{
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 2px;
            color: {TEXT_DIM};
            padding: 2px 0;
        }}

        #ToggleBtn {{
            background: transparent;
            border: 1px solid {BORDER};
            border-radius: 3px;
            color: {TEXT_DIM};
            font-size: 7px;
            padding: 0px;
        }}
        #ToggleBtn:hover {{
            background: {BORDER};
            color: {TEXT_PRI};
        }}

        #MemoryBox, #HistoryBox {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 6px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 12px;
            color: {TEXT_SEC};
            padding: 6px;
        }}

        #StatusBar {{
            background: {BG_PANEL};
            border-top: 1px solid {BORDER};
        }}

        #StatusLabel {{
            font-size: 11px;
            color: {TEXT_DIM};
        }}

        #StatusInfo {{
            font-size: 11px;
            color: {TEXT_DIM};
        }}
        """)

    # ── Visual state helper ───────────────────────────────────────────

    def _set_visual_state(self, state: str):
        self.orb.set_state(state)
        self.main_panel.sphere.set_state(state)

    # ── Sidebar toggles ───────────────────────────────────────────────

    def _toggle_sidebar(self):
        if self.sidebar.isVisible():
            sizes = self._splitter.sizes()
            self._sidebar_last_size = max(sizes[1], 280)
            self.sidebar.setVisible(False)
            total = sizes[0] + sizes[1]
            self._splitter.setSizes([total - 24, 24])
            self._sidebar_tab.setText("▶")
        else:
            self.sidebar.setVisible(True)
            total = sum(self._splitter.sizes())
            self._splitter.setSizes([total - self._sidebar_last_size,
                                     self._sidebar_last_size])
            self._sidebar_tab.setText("◀")

    def _toggle_memory(self):
        visible = not self.memory_box.isVisible()
        self.memory_box.setVisible(visible)
        self._mem_toggle.setText("▼" if visible else "▶")

    def _toggle_recent(self):
        visible = not self.history_box.isVisible()
        self.history_box.setVisible(visible)
        self._hist_toggle.setText("▼" if visible else "▶")

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_send(self):
        command = self.input_field.text().strip()
        if not command or self._worker_busy():
            return

        self.input_field.clear()
        self._clear_log()
        self._set_status(f"Processing: {command}", ACCENT_1)
        self._set_visual_state(STATE_THINKING)
        self.send_btn.setEnabled(False)
        self.response_label.setText(f"⌛  {command}")

        self._worker = AgentWorker(self.agent, command)
        self._worker.preamble.connect(self._on_preamble)
        self._worker.step_start.connect(self._on_step_start)
        self._worker.step_done.connect(self._on_step_done)
        self._worker.plan_done.connect(self._on_plan_done)
        self._worker.output_text.connect(self._on_output)
        self._worker.start()

    def _on_preamble(self, text: str):
        if text:
            self.response_label.setText(text)
            self._set_visual_state(STATE_EXECUTING)

    def _on_step_start(self, idx: int, description: str):
        while len(self._step_cards) <= idx:
            card = StepCard(len(self._step_cards), description)
            self._step_cards.append(card)
            self.log_layout.insertWidget(self.log_layout.count() - 1, card)
        self._step_cards[idx].set_running()
        self._scroll_log_to_bottom()
        self._set_status(description.rstrip(" …"), ACCENT_1)

    def _on_step_done(self, idx: int, status: str, message: str):
        if idx < len(self._step_cards):
            if status == "done":
                self._step_cards[idx].set_done(message)
            else:
                self._step_cards[idx].set_failed(message)

    def _on_plan_done(self, success: bool, summary: str):
        if success:
            self._set_visual_state(STATE_SUCCESS)
            self._set_status("Done.", SUCCESS)
        else:
            self._set_visual_state(STATE_ERROR)
            self._set_status("One or more steps failed.", ERROR)

        self.send_btn.setEnabled(True)
        self._refresh_memory()
        QTimer.singleShot(3000, lambda: self._set_visual_state(STATE_IDLE))

    def _on_output(self, text: str):
        self.output_box.setVisible(True)
        self.output_box.setPlainText(text)

    def _clear_log(self):
        for card in self._step_cards:
            self.log_layout.removeWidget(card)
            card.deleteLater()
        self._step_cards.clear()
        self.output_box.setVisible(False)
        self.output_box.clear()
        self.response_label.setText("Ready. Type a command below.")

    def _scroll_log_to_bottom(self):
        def _do_scroll():
            sb = self.log_scroll.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())
        QTimer.singleShot(50, _do_scroll)

    def _refresh_memory(self):
        try:
            mem   = self.agent.memory
            facts = mem.all_facts()
            prefs = mem.all_preferences()

            if facts or prefs:
                lines = []
                for f in facts:
                    lines.append(f"• {f['key']}: {f['value']}")
                for k, v in prefs.items():
                    if not any(f["key"] == k for f in facts):
                        lines.append(f"• {k}: {v}")
                self.memory_box.setPlainText("\n".join(lines))
            else:
                self.memory_box.setPlaceholderText("No memories yet.")

            recent = mem.recent_history(8)
            if recent:
                lines = []
                for h in recent:
                    dt    = datetime.fromtimestamp(h["created_at"]).strftime("%H:%M")
                    icon  = "✓" if h["success"] else "✗"
                    short = h["command"][:30] + ("…" if len(h["command"]) > 30 else "")
                    lines.append(f"{icon} [{dt}] {short}")
                self.history_box.setPlainText("\n".join(lines))
        except Exception:
            pass

    def _update_clock(self):
        self.clock_label.setText(datetime.now().strftime("%A  %H:%M:%S"))

    def _set_status(self, text: str, color: str = TEXT_DIM):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-size: 11px; color: {color};")

    def _worker_busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    # ── Keyboard shortcut ─────────────────────────────────────────────

    def keyPressEvent(self, a0):  # type: ignore[override]
        if a0 is not None and a0.key() == Qt.Key.Key_Escape:
            self.input_field.clear()
        super().keyPressEvent(a0)
