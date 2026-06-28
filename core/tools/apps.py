"""
Sentinel Tools — App Launcher (Windows)

Finds and launches applications using:
  1. Known path list
  2. Windows Registry
  3. PATH environment variable
  4. webbrowser.open() for URLs / websites
"""

import os
import re
import sys
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional, Dict, List

# ─────────────────────────────────────────────
# App path registry
# ─────────────────────────────────────────────

APP_LOOKUP: Dict[str, List[str]] = {
    "chrome": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ],
    "edge": [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe"),
    ],
    "firefox": [
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Mozilla Firefox\firefox.exe"),
    ],
    "brave": [
        os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
    ],
    "opera": [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Opera\launcher.exe"),
        os.path.expandvars(r"%APPDATA%\Opera Software\Opera Stable\opera.exe"),
    ],
    "visual_studio": [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\devenv.exe",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\devenv.exe",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\devenv.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\devenv.exe",
    ],
    "vscode": [
        r"C:\Program Files\Microsoft VS Code\Code.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
    ],
    "rider": [
        os.path.expandvars(r"%LOCALAPPDATA%\JetBrains\Toolbox\apps\Rider\ch-0\*\bin\rider64.exe"),
        r"C:\Program Files\JetBrains\JetBrains Rider 2024.1\bin\rider64.exe",
        r"C:\Program Files\JetBrains\JetBrains Rider 2023.3\bin\rider64.exe",
    ],
    "pycharm": [
        os.path.expandvars(r"%LOCALAPPDATA%\JetBrains\Toolbox\apps\PyCharm-P\ch-0\*\bin\pycharm64.exe"),
        r"C:\Program Files\JetBrains\PyCharm 2024.1\bin\pycharm64.exe",
    ],
    "intellij": [
        os.path.expandvars(r"%LOCALAPPDATA%\JetBrains\Toolbox\apps\IDEA-U\ch-0\*\bin\idea64.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\JetBrains\Toolbox\apps\IDEA-C\ch-0\*\bin\idea64.exe"),
    ],
    "github_desktop": [
        os.path.expandvars(r"%LOCALAPPDATA%\GitHubDesktop\GitHubDesktop.exe"),
        os.path.expandvars(r"%APPDATA%\GitHub Desktop\GitHubDesktop.exe"),
    ],
    "outlook": [
        r"C:\Program Files\Microsoft Office\root\Office16\OUTLOOK.EXE",
        r"C:\Program Files (x86)\Microsoft Office\root\Office16\OUTLOOK.EXE",
    ],
    "winword": [
        r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE",
    ],
    "excel": [
        r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
        r"C:\Program Files (x86)\Microsoft Office\root\Office16\EXCEL.EXE",
    ],
    "powerpoint": [
        r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE",
        r"C:\Program Files (x86)\Microsoft Office\root\Office16\POWERPNT.EXE",
    ],
    "onenote": [
        r"C:\Program Files\Microsoft Office\root\Office16\ONENOTE.EXE",
        r"C:\Program Files (x86)\Microsoft Office\root\Office16\ONENOTE.EXE",
    ],
    "task_manager":  ["taskmgr.exe"],
    "notepad":       ["notepad.exe"],
    "mspaint":       ["mspaint.exe"],
    "calculator":    ["calc.exe"],
    "explorer":      ["explorer.exe"],
    "cmd":           ["cmd.exe"],
    "powershell":    ["powershell.exe"],
    "snippingtool":  ["snippingtool.exe", "SnippingTool.exe"],
    "control":       ["control.exe"],
    "ms-settings":   ["ms-settings:"],
    "spotify": [
        os.path.expandvars(r"%APPDATA%\Spotify\Spotify.exe"),
    ],
    "discord": [
        os.path.expandvars(r"%LOCALAPPDATA%\Discord\Update.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Discord\app-*\Discord.exe"),
    ],
    "slack": [
        os.path.expandvars(r"%LOCALAPPDATA%\slack\slack.exe"),
    ],
    "steam": [
        r"C:\Program Files (x86)\Steam\steam.exe",
        r"C:\Program Files\Steam\steam.exe",
    ],
    "teams": [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Teams\current\Teams.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Microsoft\Teams\current\Teams.exe"),
    ],
    "zoom": [
        os.path.expandvars(r"%APPDATA%\Zoom\bin\Zoom.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Zoom\bin\Zoom.exe"),
    ],
    "obs": [
        r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
        r"C:\Program Files (x86)\obs-studio\bin\32bit\obs32.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\obs-studio\bin\64bit\obs64.exe"),
    ],
    "vlc": [
        r"C:\Program Files\VideoLAN\VLC\vlc.exe",
        r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
    ],
    "winrar": [
        r"C:\Program Files\WinRAR\WinRAR.exe",
        r"C:\Program Files (x86)\WinRAR\WinRAR.exe",
    ],
    "7zip": [
        r"C:\Program Files\7-Zip\7zFM.exe",
        r"C:\Program Files (x86)\7-Zip\7zFM.exe",
    ],
    "figma": [
        os.path.expandvars(r"%LOCALAPPDATA%\Figma\Figma.exe"),
    ],
    "blender": [
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
    ],
    "photoshop": [
        r"C:\Program Files\Adobe\Adobe Photoshop 2024\Photoshop.exe",
        r"C:\Program Files\Adobe\Adobe Photoshop 2023\Photoshop.exe",
    ],
    "premiere": [
        r"C:\Program Files\Adobe\Adobe Premiere Pro 2024\Adobe Premiere Pro.exe",
        r"C:\Program Files\Adobe\Adobe Premiere Pro 2023\Adobe Premiere Pro.exe",
    ],
    "epic_games": [
        os.path.expandvars(r"%LOCALAPPDATA%\EpicGamesLauncher\Portal\Binaries\Win64\EpicGamesLauncher.exe"),
    ],
    "battle_net": [
        r"C:\Program Files (x86)\Battle.net\Battle.net.exe",
        r"C:\Program Files\Battle.net\Battle.net.exe",
    ],
    "whatsapp": [
        os.path.expandvars(r"%LOCALAPPDATA%\WhatsApp\WhatsApp.exe"),
    ],
    "telegram": [
        os.path.expandvars(r"%APPDATA%\Telegram Desktop\Telegram.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Telegram Desktop\Telegram.exe"),
    ],
    "signal": [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\signal-desktop\Signal.exe"),
    ],
    "notion": [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Notion\Notion.exe"),
    ],
    "cursor": [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\cursor\Cursor.exe"),
    ],
    "windsurf": [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\windsurf\Windsurf.exe"),
    ],
}

# Shell-protocol URIs
SHELL_URIS: Dict[str, str] = {
    "ms-settings": "ms-settings:",
    "settings":    "ms-settings:",
}

# Website mappings — "open youtube" opens the site in the browser
WEBSITE_MAP: Dict[str, str] = {
    "youtube":    "https://www.youtube.com",
    "netflix":    "https://www.netflix.com",
    "reddit":     "https://www.reddit.com",
    "twitter":    "https://www.twitter.com",
    "x":          "https://www.x.com",
    "instagram":  "https://www.instagram.com",
    "facebook":   "https://www.facebook.com",
    "github":     "https://www.github.com",
    "google":     "https://www.google.com",
    "gmail":      "https://mail.google.com",
    "chat":       "https://chat.openai.com",
    "chatgpt":    "https://chat.openai.com",
    "claude":     "https://claude.ai",
    "wikipedia":  "https://www.wikipedia.org",
    "amazon":     "https://www.amazon.com",
    "ebay":       "https://www.ebay.com",
    "twitch":     "https://www.twitch.tv",
    "linkedin":   "https://www.linkedin.com",
    "notion":     "https://www.notion.so",
    "figma":      "https://www.figma.com",
}


class AppLauncher:

    # ── Finding ──────────────────────────────────────────────────────

    def find_app(self, app_key: str) -> Optional[str]:
        if app_key is None:
            return None
        app_key = app_key.lower().strip()

        # URL passed directly
        if app_key.startswith(("http://", "https://")):
            return app_key

        # Shell URI
        if app_key in SHELL_URIS:
            return SHELL_URIS[app_key]

        # Website shortcut
        if app_key in WEBSITE_MAP:
            return WEBSITE_MAP[app_key]

        # Built-in lookup
        for candidate in APP_LOOKUP.get(app_key, []):
            if "*" in candidate:
                import glob
                matches = [Path(p) for p in glob.glob(candidate)]
                if matches:
                    matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                    return str(matches[0])
            elif Path(candidate).exists():
                return candidate

        # Windows Registry
        if sys.platform == "win32":
            reg_path = self._find_in_registry(app_key)
            if reg_path:
                return reg_path

        # PATH
        return self._find_in_path(app_key)

    def _find_in_registry(self, app_key: str) -> Optional[str]:
        try:
            import winreg
            search_names = [app_key, app_key.replace("_", " "), app_key.title()]
            reg_paths = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths"),
            ]
            for hive, reg_base in reg_paths:
                try:
                    with winreg.OpenKey(hive, reg_base) as base:
                        i = 0
                        while True:
                            try:
                                sub_name = winreg.EnumKey(base, i)
                                i += 1
                                name_lower = sub_name.lower().replace(".exe", "")
                                if any(s in name_lower for s in search_names):
                                    with winreg.OpenKey(base, sub_name) as sub:
                                        val, _ = winreg.QueryValueEx(sub, "")
                                        if val and Path(val).exists():
                                            return val
                            except OSError:
                                break
                except OSError:
                    continue
        except ImportError:
            pass
        return None

    def _find_in_path(self, app_key: str) -> Optional[str]:
        names = [
            app_key,
            f"{app_key}.exe",
            app_key.replace("_", ""),
            f"{app_key.replace('_', '')}.exe",
        ]
        for name in names:
            for directory in os.environ.get("PATH", "").split(os.pathsep):
                full = Path(directory) / name
                if full.exists():
                    return str(full)
        return None

    # ── Launching ─────────────────────────────────────────────────────

    def launch_app(self, app_key: str, path: Optional[str] = None) -> tuple:
        if not app_key:
            return False, "No application specified."

        resolved = path or self.find_app(app_key)

        if resolved is None:
            return False, (
                f"Could not find '{app_key}'. "
                "Make sure it is installed or check the spelling."
            )

        try:
            # URL / website
            if resolved.startswith(("http://", "https://")):
                webbrowser.open(resolved)
                label = _friendly(app_key)
                return True, f"Opened {label} in your browser."

            if sys.platform == "win32":
                if resolved.startswith("ms-"):
                    subprocess.Popen(["start", resolved], shell=True)
                elif resolved.endswith(".exe"):
                    subprocess.Popen([resolved], creationflags=subprocess.DETACHED_PROCESS)
                else:
                    os.startfile(resolved)
            else:
                subprocess.Popen(
                    ["xdg-open", resolved] if sys.platform != "darwin" else ["open", resolved]
                )
            return True, f"'{_friendly(app_key)}' launched successfully."
        except Exception as ex:
            return False, f"Failed to launch '{app_key}': {ex}"

    def open_url(self, url: str) -> tuple:
        """Open an arbitrary URL in the default browser."""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            webbrowser.open(url)
            return True, f"Opened {url} in your browser."
        except Exception as e:
            return False, f"Could not open URL: {e}"

    def close_app(self, app_key: str) -> tuple:
        names = _process_names(app_key)
        killed = 0
        for name in names:
            try:
                if sys.platform == "win32":
                    result = subprocess.run(
                        ["taskkill", "/IM", name, "/F"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        killed += 1
                else:
                    result = subprocess.run(["pkill", "-f", name], capture_output=True)
                    if result.returncode == 0:
                        killed += 1
            except Exception:
                pass
        if killed:
            return True, f"Closed {_friendly(app_key)}."
        return False, f"'{app_key}' does not appear to be running."

    def web_search(self, query: str) -> tuple:
        for prefix in ["search for", "search online for", "google", "search the web for",
                       "search online", "look up", "find out about"]:
            query = re.sub(rf"^{re.escape(prefix)}\s*", "", query, flags=re.IGNORECASE).strip()
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)
        return True, f"Searching the web for '{query}'."


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

_FRIENDLY_NAMES: Dict[str, str] = {
    "chrome":         "Google Chrome",
    "edge":           "Microsoft Edge",
    "firefox":        "Firefox",
    "brave":          "Brave",
    "opera":          "Opera",
    "visual_studio":  "Visual Studio",
    "vscode":         "VS Code",
    "rider":          "JetBrains Rider",
    "pycharm":        "PyCharm",
    "intellij":       "IntelliJ IDEA",
    "github_desktop": "GitHub Desktop",
    "outlook":        "Microsoft Outlook",
    "task_manager":   "Task Manager",
    "notepad":        "Notepad",
    "mspaint":        "Paint",
    "calculator":     "Calculator",
    "explorer":       "File Explorer",
    "winword":        "Microsoft Word",
    "excel":          "Microsoft Excel",
    "powerpoint":     "PowerPoint",
    "onenote":        "OneNote",
    "spotify":        "Spotify",
    "discord":        "Discord",
    "slack":          "Slack",
    "steam":          "Steam",
    "teams":          "Microsoft Teams",
    "zoom":           "Zoom",
    "obs":            "OBS Studio",
    "vlc":            "VLC",
    "winrar":         "WinRAR",
    "7zip":           "7-Zip",
    "figma":          "Figma",
    "blender":        "Blender",
    "photoshop":      "Photoshop",
    "premiere":       "Premiere Pro",
    "epic_games":     "Epic Games Launcher",
    "battle_net":     "Battle.net",
    "whatsapp":       "WhatsApp",
    "telegram":       "Telegram",
    "signal":         "Signal",
    "notion":         "Notion",
    "cmd":            "Command Prompt",
    "powershell":     "PowerShell",
    "snippingtool":   "Snipping Tool",
    "control":        "Control Panel",
    "ms-settings":    "Settings",
    "cursor":         "Cursor",
    "windsurf":       "Windsurf",
    # websites
    "youtube":    "YouTube",
    "netflix":    "Netflix",
    "reddit":     "Reddit",
    "twitter":    "Twitter / X",
    "instagram":  "Instagram",
    "facebook":   "Facebook",
    "github":     "GitHub",
    "google":     "Google",
    "gmail":      "Gmail",
    "chatgpt":    "ChatGPT",
    "claude":     "Claude",
    "wikipedia":  "Wikipedia",
    "amazon":     "Amazon",
    "twitch":     "Twitch",
    "linkedin":   "LinkedIn",
}


def _friendly(app_key: str) -> str:
    return _FRIENDLY_NAMES.get(app_key, app_key.replace("_", " ").title())


_PROCESS_MAP: Dict[str, List[str]] = {
    "chrome":         ["chrome.exe"],
    "edge":           ["msedge.exe"],
    "firefox":        ["firefox.exe"],
    "brave":          ["brave.exe"],
    "visual_studio":  ["devenv.exe"],
    "vscode":         ["Code.exe"],
    "rider":          ["rider64.exe"],
    "pycharm":        ["pycharm64.exe"],
    "github_desktop": ["GitHubDesktop.exe"],
    "outlook":        ["OUTLOOK.EXE"],
    "task_manager":   ["Taskmgr.exe"],
    "notepad":        ["notepad.exe"],
    "mspaint":        ["mspaint.exe"],
    "calculator":     ["Calculator.exe", "calc.exe"],
    "spotify":        ["Spotify.exe"],
    "discord":        ["Discord.exe"],
    "slack":          ["slack.exe"],
    "steam":          ["steam.exe"],
    "teams":          ["Teams.exe"],
    "zoom":           ["Zoom.exe"],
    "obs":            ["obs64.exe", "obs32.exe"],
    "vlc":            ["vlc.exe"],
    "winrar":         ["WinRAR.exe"],
    "7zip":           ["7zFM.exe"],
    "whatsapp":       ["WhatsApp.exe"],
    "telegram":       ["Telegram.exe"],
    "signal":         ["Signal.exe"],
    "cmd":            ["cmd.exe"],
    "powershell":     ["powershell.exe"],
    "notion":         ["Notion.exe"],
    "figma":          ["Figma.exe"],
    "cursor":         ["Cursor.exe"],
}


def _process_names(app_key: str) -> List[str]:
    return _PROCESS_MAP.get(app_key, [f"{app_key}.exe"])
