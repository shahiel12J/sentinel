"""
Sentinel Tools — System Control (Windows)

Handles:
  - volume_up / volume_down / set_volume / toggle_mute  (ctypes VK + pycaw)
  - media play / pause / next / prev / stop             (ctypes VK)
  - screenshot                                          (Pillow)
  - shutdown / restart / sleep / lock / minimize_all    (subprocess + ctypes)
  - brightness_up / brightness_down                     (WMI or PowerShell)
  - system_info / list_processes                        (psutil)
  - time / date query                                   (datetime)
  - schedule_reminder                                   (background thread)
  - show_help
"""

import sys
import os
import time
import ctypes
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List

import psutil

# ─────────────────────────────────────────────
# Virtual key codes (Windows)
# ─────────────────────────────────────────────

_VK_VOLUME_MUTE      = 0xAD
_VK_VOLUME_DOWN      = 0xAE
_VK_VOLUME_UP        = 0xAF
_VK_MEDIA_NEXT       = 0xB0
_VK_MEDIA_PREV       = 0xB1
_VK_MEDIA_STOP       = 0xB2
_VK_MEDIA_PLAY_PAUSE = 0xB3
_VK_LWIN             = 0x5B
_VK_D                = 0x44
_VK_L                = 0x4C
_KEYEVENTF_KEYUP     = 0x0002

SCREENSHOT_DIR = Path.home() / "Pictures" / "Sentinel Screenshots"


class SystemTools:

    def __init__(self):
        self._screenshot_buffer = None

    # ── Internal key sender ───────────────────────────────────────────

    def _press(self, vk: int, times: int = 1) -> bool:
        """Send a global virtual keypress using ctypes (no external deps)."""
        if sys.platform != "win32":
            return False
        try:
            for _ in range(times):
                ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
                ctypes.windll.user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False

    # ── Volume ────────────────────────────────────────────────────────

    def volume_up(self, step: float = 0.10) -> Tuple[bool, str]:
        return self._adjust_volume(step)

    def volume_down(self, step: float = 0.10) -> Tuple[bool, str]:
        return self._adjust_volume(-step)

    def set_volume(self, level: int) -> Tuple[bool, str]:
        """Set volume to an exact percentage (0-100). Requires pycaw for precision."""
        level = max(0, min(100, level))
        if sys.platform != "win32":
            return False, "Volume control is Windows-only."
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            import comtypes
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)  # type: ignore[union-attr, attr-defined]
            volume = interface.QueryInterface(IAudioEndpointVolume)
            volume.SetMasterVolumeLevelScalar(level / 100.0, None)
            return True, f"Volume set to {level}%."
        except ImportError:
            pass
        except Exception:
            pass
        # Fallback: approximate via key presses from current level
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            import comtypes
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)  # type: ignore[union-attr, attr-defined]
            volume = interface.QueryInterface(IAudioEndpointVolume)
            current = int(volume.GetMasterVolumeLevelScalar() * 100)
            diff = level - current
            if diff > 0:
                self._press(_VK_VOLUME_UP, max(1, diff // 2))
            elif diff < 0:
                self._press(_VK_VOLUME_DOWN, max(1, abs(diff) // 2))
            return True, f"Volume adjusted toward {level}%."
        except Exception:
            pass
        return False, "pycaw is not installed. Run: pip install pycaw"

    def toggle_mute(self) -> Tuple[bool, str]:
        if sys.platform != "win32":
            return False, "Volume control is Windows-only."
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            import comtypes
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)  # type: ignore[union-attr, attr-defined]
            volume = interface.QueryInterface(IAudioEndpointVolume)
            muted = volume.GetMute()
            volume.SetMute(not muted, None)
            return True, f"Audio {'muted' if not muted else 'unmuted'}."
        except Exception:
            pass
        # Fallback: mute key via ctypes
        if self._press(_VK_VOLUME_MUTE):
            return True, "Mute toggled."
        return False, "Could not control volume."

    def _adjust_volume(self, delta: float) -> Tuple[bool, str]:
        if sys.platform != "win32":
            return False, "Volume control is Windows-only."
        # Try pycaw for exact percentage
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            import comtypes
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)  # type: ignore[union-attr, attr-defined]
            volume = interface.QueryInterface(IAudioEndpointVolume)
            current = volume.GetMasterVolumeLevelScalar()
            new_vol = min(1.0, max(0.0, current + delta))
            volume.SetMasterVolumeLevelScalar(new_vol, None)
            direction = "increased" if delta > 0 else "decreased"
            return True, f"Volume {direction} to {int(new_vol * 100)}%."
        except Exception:
            pass
        # Fallback: global keybd_event — always works on Windows, no deps
        vk    = _VK_VOLUME_UP if delta > 0 else _VK_VOLUME_DOWN
        steps = max(1, round(abs(delta) * 50))   # Windows default: ~2% per step
        if self._press(vk, steps):
            direction = "increased" if delta > 0 else "decreased"
            return True, f"Volume {direction}."
        return False, "Volume control failed."

    # ── Media controls ────────────────────────────────────────────────

    def media_play_pause(self) -> Tuple[bool, str]:
        if self._press(_VK_MEDIA_PLAY_PAUSE):
            return True, "Play/Pause toggled."
        return False, "Media control unavailable."

    def media_next(self) -> Tuple[bool, str]:
        if self._press(_VK_MEDIA_NEXT):
            return True, "Skipped to next track."
        return False, "Media control unavailable."

    def media_prev(self) -> Tuple[bool, str]:
        if self._press(_VK_MEDIA_PREV):
            return True, "Went back to previous track."
        return False, "Media control unavailable."

    def media_stop(self) -> Tuple[bool, str]:
        if self._press(_VK_MEDIA_STOP):
            return True, "Media stopped."
        return False, "Media control unavailable."

    # ── Screenshot ────────────────────────────────────────────────────

    def screenshot(self) -> Tuple[bool, str]:
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            self._screenshot_buffer = img
            return True, "Screen captured."
        except ImportError:
            return False, "Pillow is required: pip install Pillow"
        except Exception as e:
            return False, f"Screenshot failed: {e}"

    def save_screenshot(self) -> Tuple[bool, str]:
        if self._screenshot_buffer is None:
            ok, msg = self.screenshot()
            if not ok:
                return False, msg
        assert self._screenshot_buffer is not None
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = SCREENSHOT_DIR / f"sentinel_{ts}.png"
        try:
            self._screenshot_buffer.save(str(path))
            self._screenshot_buffer = None
            return True, f"Screenshot saved to {path}."
        except Exception as e:
            return False, f"Could not save screenshot: {e}"

    # ── Shutdown / restart / sleep / lock ────────────────────────────

    def shutdown(self, minutes: int = 0) -> Tuple[bool, str]:
        if sys.platform == "win32":
            seconds = minutes * 60 if minutes else 30
            subprocess.Popen(["shutdown", "/s", "/t", str(seconds)])
            if minutes:
                return True, f"Shutdown scheduled in {minutes} minute{'s' if minutes != 1 else ''}."
            return True, "Shutdown initiated (30-second countdown)."
        subprocess.Popen(["sudo", "shutdown", "-h", f"+{minutes}"])
        return True, f"Shutdown scheduled in {minutes} minutes."

    def cancel_shutdown(self) -> Tuple[bool, str]:
        if sys.platform == "win32":
            subprocess.Popen(["shutdown", "/a"])
            return True, "Shutdown cancelled."
        return False, "Cannot cancel on this platform."

    def restart(self, minutes: int = 0) -> Tuple[bool, str]:
        if sys.platform == "win32":
            seconds = minutes * 60
            subprocess.Popen(["shutdown", "/r", "/t", str(seconds)])
            if minutes:
                return True, f"Restart scheduled in {minutes} minute{'s' if minutes != 1 else ''}."
            return True, "Restarting now."
        subprocess.Popen(["sudo", "reboot"])
        return True, "Restarting."

    def sleep(self) -> Tuple[bool, str]:
        if sys.platform == "win32":
            subprocess.Popen(["powershell", "-c",
                "Add-Type -Assembly System.Windows.Forms; "
                "[System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)"])
            return True, "Initiating sleep mode."
        elif sys.platform == "darwin":
            subprocess.Popen(["pmset", "sleepnow"])
            return True, "Initiating sleep mode."
        subprocess.Popen(["systemctl", "suspend"])
        return True, "Initiating sleep mode."

    def lock_screen(self) -> Tuple[bool, str]:
        if sys.platform == "win32":
            ctypes.windll.user32.LockWorkStation()
            return True, "Screen locked."
        elif sys.platform == "darwin":
            subprocess.Popen(["pmset", "displaysleepnow"])
            return True, "Screen locked."
        subprocess.Popen(["loginctl", "lock-session"])
        return True, "Screen locked."

    # ── Window management ─────────────────────────────────────────────

    def minimize_all(self) -> Tuple[bool, str]:
        """Show desktop (Win+D)."""
        if sys.platform != "win32":
            return False, "Window management is Windows-only."
        try:
            ctypes.windll.user32.keybd_event(_VK_LWIN, 0, 0, 0)
            ctypes.windll.user32.keybd_event(_VK_D, 0, 0, 0)
            ctypes.windll.user32.keybd_event(_VK_D, 0, _KEYEVENTF_KEYUP, 0)
            ctypes.windll.user32.keybd_event(_VK_LWIN, 0, _KEYEVENTF_KEYUP, 0)
            return True, "Desktop shown (all windows minimized)."
        except Exception as e:
            return False, f"Could not minimize windows: {e}"

    # ── Brightness ────────────────────────────────────────────────────

    def brightness_up(self, step: int = 10) -> Tuple[bool, str]:
        return self._adjust_brightness(step)

    def brightness_down(self, step: int = 10) -> Tuple[bool, str]:
        return self._adjust_brightness(-step)

    def _adjust_brightness(self, delta: int) -> Tuple[bool, str]:
        if sys.platform != "win32":
            return False, "Brightness control is Windows-only."
        try:
            import wmi  # type: ignore[import]
            wmi_obj = wmi.WMI(namespace="wmi")
            methods = wmi_obj.WmiMonitorBrightnessMethods()
            brightness = wmi_obj.WmiMonitorBrightness()
            if methods and brightness:
                current = brightness[0].CurrentBrightness
                new_val = max(0, min(100, current + delta))
                methods[0].WmiSetBrightness(Brightness=new_val, Timeout=0)
                direction = "increased" if delta > 0 else "decreased"
                return True, f"Brightness {direction} to {new_val}%."
        except ImportError:
            pass
        except Exception:
            pass
        # PowerShell fallback
        try:
            script = (
                "$b=(Get-WmiObject -NS root/WMI -Class WmiMonitorBrightness).CurrentBrightness;"
                f"$n=[Math]::Max(0,[Math]::Min(100,$b+({delta})));"
                "(Get-WmiObject -NS root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,$n)"
            )
            result = subprocess.run(
                ["powershell", "-c", script],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                direction = "increased" if delta > 0 else "decreased"
                return True, f"Brightness {direction}."
            return False, "Brightness control failed (may not be supported on desktop monitors)."
        except Exception as e:
            return False, f"Brightness control failed: {e}"

    # ── System info ───────────────────────────────────────────────────

    def system_info(self) -> Tuple[bool, str]:
        try:
            cpu_pct  = psutil.cpu_percent(interval=0.5)
            cpu_freq = psutil.cpu_freq()
            ram      = psutil.virtual_memory()
            disk     = psutil.disk_usage("/")
            net      = psutil.net_io_counters()

            lines = [
                "═══ System Status ═══",
                "",
                f"🖥  CPU Usage    : {cpu_pct:.1f}%"
                + (f"  @ {cpu_freq.current:.0f} MHz" if cpu_freq else ""),
                f"💾  RAM          : {_human(ram.used)} / {_human(ram.total)}"
                + f"  ({ram.percent:.1f}% used)",
                f"💿  Disk (/)     : {_human(disk.used)} / {_human(disk.total)}"
                + f"  ({disk.percent:.1f}% used)",
                f"🌐  Network ↑    : {_human(net.bytes_sent)} sent",
                f"           ↓    : {_human(net.bytes_recv)} received",
            ]
            try:
                batt = psutil.sensors_battery()
                if batt:
                    plugged = "⚡ Charging" if batt.power_plugged else "🔋 On battery"
                    lines.append(f"🔋  Battery      : {batt.percent:.0f}%  ({plugged})")
            except Exception:
                pass
            try:
                procs = sorted(
                    (p for p in psutil.process_iter(["pid", "name", "cpu_percent"])
                     if p.info["cpu_percent"] is not None),
                    key=lambda p: p.info["cpu_percent"],
                    reverse=True
                )[:5]
                if procs:
                    lines += ["", "Top Processes:"]
                    for p in procs:
                        lines.append(f"  {p.info['name']:<30} CPU: {p.info['cpu_percent']:.1f}%")
            except Exception:
                pass
            return True, "\n".join(lines)
        except Exception as e:
            return False, f"Could not retrieve system info: {e}"

    def list_processes(self) -> Tuple[bool, str]:
        """Return a list of running user-visible processes."""
        try:
            seen: set = set()
            lines = ["Running processes:"]
            procs = sorted(
                psutil.process_iter(["name", "pid", "memory_percent"]),
                key=lambda p: p.info.get("memory_percent") or 0,
                reverse=True
            )
            for p in procs[:25]:
                name = p.info.get("name", "?")
                if name and name.lower() not in seen and not name.startswith("svchost"):
                    seen.add(name.lower())
                    mem = p.info.get("memory_percent") or 0
                    lines.append(f"  {name:<35}  MEM: {mem:.1f}%")
            return True, "\n".join(lines)
        except Exception as e:
            return False, f"Could not list processes: {e}"

    # ── Time / date ───────────────────────────────────────────────────

    def get_time_date(self) -> Tuple[bool, str]:
        now = datetime.now()
        return True, (
            f"🕐  Time : {now.strftime('%H:%M:%S')}\n"
            f"📅  Date : {now.strftime('%A, %d %B %Y')}"
        )

    # ── Schedule reminder ─────────────────────────────────────────────

    def schedule_reminder(
        self,
        minutes: int,
        message: str = "",
        callback=None,
    ) -> Tuple[bool, str]:
        if minutes <= 0:
            return False, "Please specify a positive number of minutes."

        def _fire():
            time.sleep(minutes * 60)
            reminder_msg = f"Sentinel Reminder\n\n{message or 'Time is up!'}"
            if callback:
                callback(reminder_msg)
            elif sys.platform == "win32":
                try:
                    from plyer import notification  # type: ignore[import]
                    notification.notify(
                        title="Sentinel Reminder",
                        message=message or "Your reminder!",
                        app_name="Sentinel",
                        timeout=10,
                    )
                except ImportError:
                    subprocess.Popen(
                        ["powershell", "-c",
                         f'[System.Windows.MessageBox]::Show("{reminder_msg}", "Sentinel")']
                    )

        threading.Thread(target=_fire, daemon=True).start()
        return True, f"Reminder set for {minutes} minute{'s' if minutes != 1 else ''}."

    # ── Help ─────────────────────────────────────────────────────────

    def show_help(self) -> Tuple[bool, str]:
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║                    SENTINEL COMMANDS                         ║
╠══════════════════════════════════════════════════════════════╣
║ APP CONTROL                                                  ║
║   open chrome / rider / spotify / discord / edge             ║
║   close chrome / visual studio                               ║
║   open youtube.com  |  go to reddit.com                      ║
║                                                              ║
║ VOLUME & MEDIA                                               ║
║   volume up / down  |  mute / unmute                         ║
║   set volume to 50  |  max volume                            ║
║   play / pause  |  next song  |  previous track              ║
║                                                              ║
║ SCREEN & SYSTEM                                              ║
║   take a screenshot                                          ║
║   brightness up / down                                       ║
║   minimize all windows  |  show desktop                      ║
║   lock screen  |  sleep                                      ║
║   shutdown in 30 minutes  |  restart                         ║
║   what time is it  |  what's today's date                    ║
║                                                              ║
║ FILES                                                        ║
║   find all pdfs  |  search for mp3s in downloads             ║
║   create a folder called Demo                                ║
║   move every png into Images  |  list files in desktop       ║
║                                                              ║
║ MEMORY                                                       ║
║   remember my IDE is Rider                                   ║
║   what is my favourite ide?                                  ║
║                                                              ║
║ INFO                                                         ║
║   system info  |  what's running  |  remind me in 10 min     ║
╚══════════════════════════════════════════════════════════════╝
""".strip()
        return True, help_text


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} PB"
