"""
Sentinel Intent Classifier

Hybrid classifier:
  1. Neural  — SentinelTransformer (used when model is trained)
  2. Rule-based fallback — regex + keyword matching (always available)

Returns a ClassificationResult with intent, confidence, and extracted entities.
"""

import re
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch

# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────

@dataclass
class ClassificationResult:
    intent:     str
    confidence: float
    entities:   Dict[str, str] = field(default_factory=dict)
    source:     str = "rule"       # "neural" | "rule"

    def __repr__(self) -> str:
        return (f"ClassificationResult(intent={self.intent!r}, "
                f"conf={self.confidence:.2f}, entities={self.entities}, "
                f"src={self.source})")


# ─────────────────────────────────────────────
# Known app registry
# ─────────────────────────────────────────────

APP_ALIASES: Dict[str, str] = {
    # Browsers
    "chrome":               "chrome",
    "google chrome":        "chrome",
    "browser":              "chrome",
    "edge":                 "edge",
    "microsoft edge":       "edge",
    "firefox":              "firefox",
    "mozilla":              "firefox",
    "brave":                "brave",
    "opera":                "opera",
    # IDEs
    "visual studio":        "visual_studio",
    "vs":                   "visual_studio",
    "vscode":               "vscode",
    "visual studio code":   "vscode",
    "vs code":              "vscode",
    "code":                 "vscode",
    "rider":                "rider",
    "jetbrains rider":      "rider",
    "pycharm":              "pycharm",
    "intellij":             "intellij",
    "idea":                 "intellij",
    "cursor":               "cursor",
    "windsurf":             "windsurf",
    # Productivity
    "github desktop":       "github_desktop",
    "outlook":              "outlook",
    "microsoft outlook":    "outlook",
    "email":                "outlook",
    "my email":             "outlook",
    "word":                 "winword",
    "microsoft word":       "winword",
    "excel":                "excel",
    "microsoft excel":      "excel",
    "powerpoint":           "powerpoint",
    "power point":          "powerpoint",
    "onenote":              "onenote",
    "one note":             "onenote",
    "teams":                "teams",
    "microsoft teams":      "teams",
    "notion":               "notion",
    "zoom":                 "zoom",
    # Social / comms
    "discord":              "discord",
    "slack":                "slack",
    "whatsapp":             "whatsapp",
    "telegram":             "telegram",
    "signal":               "signal",
    # Media / entertainment
    "spotify":              "spotify",
    "steam":                "steam",
    "epic games":           "epic_games",
    "epic":                 "epic_games",
    "battle net":           "battle_net",
    "blizzard":             "battle_net",
    "vlc":                  "vlc",
    "vlc media player":     "vlc",
    "obs":                  "obs",
    "obs studio":           "obs",
    "twitch":               "twitch",           # website
    "youtube":              "youtube",          # website
    "netflix":              "netflix",          # website
    # Websites
    "reddit":               "reddit",
    "twitter":              "twitter",
    "x":                    "x",
    "instagram":            "instagram",
    "facebook":             "facebook",
    "gmail":                "gmail",
    "chatgpt":              "chatgpt",
    "chat gpt":             "chatgpt",
    "claude":               "claude",
    "wikipedia":            "wikipedia",
    "amazon":               "amazon",
    "github":               "github",
    "linkedin":             "linkedin",
    "figma":                "figma",
    # System
    "task manager":         "task_manager",
    "notepad":              "notepad",
    "paint":                "mspaint",
    "ms paint":             "mspaint",
    "calculator":           "calculator",
    "calc":                 "calculator",
    "file explorer":        "explorer",
    "explorer":             "explorer",
    "terminal":             "cmd",
    "command prompt":       "cmd",
    "cmd":                  "cmd",
    "powershell":           "powershell",
    "snipping tool":        "snippingtool",
    "snip":                 "snippingtool",
    "settings":             "ms-settings",
    "windows settings":     "ms-settings",
    "control panel":        "control",
    "winrar":               "winrar",
    "7zip":                 "7zip",
    "7-zip":                "7zip",
    "blender":              "blender",
    "photoshop":            "photoshop",
    "premiere":             "premiere",
    "premiere pro":         "premiere",
    # Memory tokens
    "ide":                  "__remembered_ide__",
    "my ide":               "__remembered_ide__",
    "my project":           "__remembered_project__",
    "my favourite ide":     "__remembered_ide__",
    "my favorite ide":      "__remembered_ide__",
}


def resolve_app(raw: str) -> str:
    """Normalise raw app string to canonical app key."""
    key = raw.strip().lower()
    return APP_ALIASES.get(key, key.replace(" ", "_"))


# ─────────────────────────────────────────────
# Rule-based entity extractors
# ─────────────────────────────────────────────

# File extension patterns
_EXT_PAT = re.compile(
    r"\b(pdf|png|jpg|jpeg|gif|txt|log|csv|xlsx|docx|mp3|mp4|exe|zip|py|json|md|html|xml)\b",
    re.IGNORECASE
)

# Folder name after "called / named"
_FOLDER_PAT = re.compile(
    r"(?:called|named|named\s+as|name\s+it)\s+['\"]?([A-Za-z0-9_\-. ]+?)['\"]?(?:\s|$)",
    re.IGNORECASE
)

# "in X minutes/hours/seconds"
_TIME_PAT = re.compile(
    r"in\s+(\d+)\s*(minute|min|hour|hr|second|sec)s?\b",
    re.IGNORECASE
)

# Path patterns
_PATH_PAT = re.compile(r"[A-Za-z]:[\\\/][^\s,;\"']+", re.IGNORECASE)

# "to X folder / into X"
_DEST_PAT = re.compile(
    r"(?:to|into)\s+(?:the\s+)?['\"]?([A-Za-z0-9_\-. ]+?)['\"]?\s*(?:folder|directory|dir|$)",
    re.IGNORECASE
)

# Preference value: "my X is Y"
_PREF_KEY_PAT   = re.compile(r"my\s+([a-z ]+?)\s+is\b", re.IGNORECASE)
_PREF_VALUE_PAT = re.compile(r"\bis\s+([a-zA-Z0-9_\-/.\\: ]+?)(?:\s*$|,|\.|;)", re.IGNORECASE)

# "open / close / launch X"
_APP_PAT = re.compile(
    r"(?:open|launch|start|close|quit|kill|terminate|exit|run|fire up|bring up)\s+"
    r"(?:up\s+)?(?:my\s+)?([a-z ]+?)(?:\s+for me)?$",
    re.IGNORECASE
)

# Filename
_FILE_PAT = re.compile(r"[\w\-]+\.\w{2,5}", re.IGNORECASE)


# URL pattern (domain.tld or full URL)
_URL_PAT = re.compile(
    r"(?:https?://)?(?:www\.)?([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})(?:/\S*)?",
    re.IGNORECASE
)

# Volume level: "50", "50%", "half", "max", "min"
_VOL_LEVEL_PAT = re.compile(r"\b(\d{1,3})\s*%?", re.IGNORECASE)


def _extract_entities(text: str, intent: str) -> Dict[str, str]:
    t = text.strip()
    entities: Dict[str, str] = {}

    if intent in ("open_app", "close_app"):
        m = _APP_PAT.search(t)
        if m:
            raw = m.group(1).strip()
            entities["app"] = resolve_app(raw)

    if intent == "open_url":
        # Try to extract a domain/URL from the text
        for prefix in ("go to", "open", "visit", "navigate to", "load"):
            cleaned = re.sub(rf"^{prefix}\s*", "", t, flags=re.IGNORECASE).strip()
            if cleaned != t:
                t_url = cleaned
                break
        else:
            t_url = t
        # Check if it looks like a URL
        m = _URL_PAT.search(t_url)
        if m:
            url = m.group(0)
            if not url.startswith("http"):
                url = "https://" + url
            entities["url"] = url
        else:
            # Try as a site name from APP_ALIASES or WEBSITE_MAP
            clean = t_url.strip().lower()
            entities["url"] = clean   # pass through for app launcher to resolve

    if intent == "file_search":
        m = _EXT_PAT.search(t)
        if m:
            entities["extension"] = m.group(1).lower()
        if re.search(r"\btoday\b", t, re.I):
            entities["time_filter"] = "today"
        elif re.search(r"\bthis week\b", t, re.I):
            entities["time_filter"] = "week"
        m = re.search(r"(?:in|from|inside)\s+(?:my\s+)?([a-z]+)\s*folder", t, re.I)
        if m:
            entities["location"] = m.group(1).lower()
        # Name-based search: "find resume", "locate budget", "where is report.pdf"
        if not entities.get("extension"):
            _name_m = re.search(
                r"(?:find|locate|where(?:'?s| is)|search for|find my|locate my|look for)"
                r"\s+(?:(?:the|my|a|an)\s+)?"
                r"([a-zA-Z0-9_\-][a-zA-Z0-9_\-\. ]*?)"
                r"(?:\s+(?:file|files|document|doc|in|on|from|today|this|folder)|\s*$)",
                t, re.I
            )
            if _name_m:
                _name = _name_m.group(1).strip()
                _generic = {
                    "files", "all", "every", "documents", "folders",
                    "pdfs", "images", "videos", "music", "photos",
                }
                if _name and _name.lower() not in _generic:
                    entities["filename"] = _name

    if intent == "create_folder":
        m = _FOLDER_PAT.search(t)
        if m:
            entities["name"] = m.group(1).strip()

    if intent in ("move_files", "copy_files"):
        m = _EXT_PAT.search(t)
        if m:
            entities["extension"] = m.group(1).lower()
        d = _DEST_PAT.search(t)
        if d:
            entities["destination"] = d.group(1).strip()

    if intent == "summarize_file":
        m = _FILE_PAT.search(t)
        if m:
            entities["filename"] = m.group(0)

    if intent == "open_file":
        m = _FILE_PAT.search(t)
        if m:
            entities["filename"] = m.group(0)

    if intent in ("shutdown", "restart", "schedule_task"):
        m = _TIME_PAT.search(t)
        if m:
            qty  = int(m.group(1))
            unit = m.group(2).lower()
            if "hour" in unit or unit == "hr":
                qty *= 60
            elif "second" in unit or unit == "sec":
                qty = max(1, qty // 60)
            entities["minutes"] = str(qty)

    if intent == "remember":
        k = _PREF_KEY_PAT.search(t)
        v = _PREF_VALUE_PAT.search(t)
        if k:
            entities["key"] = k.group(1).strip().lower()
        if v:
            entities["value"] = v.group(1).strip()

    if intent == "list_files":
        m = re.search(r"(?:in|from|on|inside)\s+(?:my\s+)?([a-z]+)\s*(?:folder|directory)?", t, re.I)
        if m:
            entities["location"] = m.group(1).lower()

    if intent == "set_volume":
        # "set volume to 50", "volume at 50%", "max volume", "half volume"
        t_low = t.lower()
        if re.search(r"\bmax\b|\bfull\b|\b100\b", t_low):
            entities["level"] = "100"
        elif re.search(r"\bmin\b|\bzero\b|\boff\b|\b0\b", t_low):
            entities["level"] = "0"
        elif re.search(r"\bhalf\b|\b50\b", t_low):
            entities["level"] = "50"
        else:
            m = _VOL_LEVEL_PAT.search(t)
            if m:
                entities["level"] = m.group(1)

    if intent == "brightness_up":
        m = _VOL_LEVEL_PAT.search(t)
        entities["step"] = m.group(1) if m else "10"

    if intent == "brightness_down":
        m = _VOL_LEVEL_PAT.search(t)
        entities["step"] = m.group(1) if m else "10"

    return entities


# ─────────────────────────────────────────────
# Rule-based classifier (always available)
# ─────────────────────────────────────────────

_RULES: List[Tuple[str, List[str]]] = [
    # ── Apps ──────────────────────────────────────────────────────────
    ("open_app",      ["open ", "launch ", "start ", "fire up", "bring up", "run "]),
    ("close_app",     ["close ", "quit ", "kill ", "terminate ", "force quit", "shut down "]),
    # ── Files ─────────────────────────────────────────────────────────
    ("file_search",   ["find files", "find all", "search for files", "search my files",
                       "search my ", "locate files", "find every", "find my ",
                       "search for .*\\.\\w+", "where are my"]),
    ("create_folder", ["create a folder", "make a folder", "new folder", "create folder",
                       "make a directory", "create a directory", "mkdir"]),
    ("move_files",    ["move ", "relocate "]),
    ("copy_files",    ["copy ", "duplicate "]),
    ("delete_files",  ["delete ", "remove ", "trash ", "erase "]),
    ("rename_file",   ["rename "]),
    ("open_file",     ["open file", "open the file"]),
    ("summarize_file",["summarize", "summarise", "tldr", "what does this file", "give me a summary"]),
    ("list_files",    ["list files", "show me files", "show files", "list all files",
                       "what files", "show all files"]),
    # ── Memory ────────────────────────────────────────────────────────
    ("remember",      ["remember ", "remember that", "keep in mind", "save this",
                       "note that", "store this", "don't forget"]),
    ("recall",        ["what is my", "what's my", "do you remember", "what did i say",
                       "recall my", "what have i told you", "what do you know about me"]),
    # ── Volume ────────────────────────────────────────────────────────
    ("set_volume",    ["set volume to", "set the volume to", "volume to ", "volume at ",
                       "set volume at", "max volume", "minimum volume", "full volume",
                       "half volume", "volume 5", "volume 1", "volume 2", "volume 3",
                       "volume 4", "volume 6", "volume 7", "volume 8", "volume 9",
                       "volume 0"]),
    ("volume_up",     ["volume up", "turn up the volume", "turn the volume up",
                       "increase volume", "increase the volume", "louder", "raise the volume",
                       "raise volume", "increase sound", "crank it up", "make it louder",
                       "a bit louder", "bit louder"]),
    ("volume_down",   ["volume down", "turn down the volume", "turn the volume down",
                       "decrease volume", "decrease the volume", "quieter", "lower the volume",
                       "lower volume", "reduce volume", "reduce the volume", "keep it down",
                       "make it quieter", "a bit quieter", "bit quieter", "less loud"]),
    ("mute",          ["mute", "unmute", "silence", "turn off sound", "toggle mute",
                       "no sound", "go quiet"]),
    # ── Media ─────────────────────────────────────────────────────────
    ("media_play",    ["play music", "play song", "resume music", "resume playback",
                       "unpause music", "unpause song", "play spotify", "play audio"]),
    ("media_pause",   ["pause music", "pause song", "pause playback", "pause the music",
                       "stop the music", "pause spotify"]),
    ("media_next",    ["next song", "next track", "skip song", "skip track", "skip this",
                       "next music", "skip ahead"]),
    ("media_prev",    ["previous song", "previous track", "prev song", "last song",
                       "go back song", "back a track"]),
    ("media_stop",    ["stop music", "stop playback", "stop spotify", "stop playing"]),
    # ── Screenshot ────────────────────────────────────────────────────
    ("screenshot",    ["screenshot", "screen capture", "screengrab", "capture the screen",
                       "print screen", "snap the screen", "take a screenshot", "grab screen"]),
    # ── Power / session ───────────────────────────────────────────────
    ("shutdown",      ["shutdown", "shut down", "power off", "turn off the computer",
                       "turn off my computer", "turn off my pc"]),
    ("restart",       ["restart", "reboot", "restart my computer", "restart the pc"]),
    ("sleep",         ["sleep", "hibernate", "put to sleep", "suspend"]),
    ("lock_screen",   ["lock", "lock screen", "lock my computer", "lock the screen",
                       "lock pc", "lock my pc", "win+l"]),
    ("cancel_shutdown",["cancel shutdown", "abort shutdown", "stop shutdown", "undo shutdown"]),
    # ── Display / windows ─────────────────────────────────────────────
    ("brightness_up", ["brightness up", "brighter", "increase brightness",
                       "screen brighter", "make it brighter", "more brightness"]),
    ("brightness_down",["brightness down", "dimmer", "decrease brightness",
                        "screen dimmer", "make it dimmer", "less brightness", "dim the screen"]),
    ("minimize_all",  ["minimize all", "show desktop", "hide all windows", "win+d",
                       "clear the screen", "minimise all"]),
    # ── System info ───────────────────────────────────────────────────
    ("system_info",   ["cpu usage", "ram usage", "memory usage", "disk space",
                       "system info", "system stats", "ip address", "battery level",
                       "how much ram", "system status", "pc info", "computer info"]),
    ("process_list",  ["running apps", "running processes", "what's running", "what is running",
                       "open apps", "active apps", "list processes", "show processes",
                       "task list", "what programs are open", "list running"]),
    # ── Web & URLs ────────────────────────────────────────────────────
    ("web_search",    ["search online", "search the web", "look up online",
                       "google it", "search google", "web search", "online search",
                       "find online", "look it up"]),
    ("open_url",      ["go to ", "visit ", "open website", "navigate to ", "load website",
                       "open url", "browse to"]),
    # ── Time / date ───────────────────────────────────────────────────
    ("time_query",    ["what time is it", "what's the time", "current time",
                       "tell me the time", "what day is it", "what's today",
                       "what is today", "today's date", "what date is it",
                       "what's the date", "current date"]),
    # ── Reminders ─────────────────────────────────────────────────────
    ("schedule_task", ["remind me", "set a timer", "alarm in", "set an alarm",
                       "set a reminder", "timer for", "notify me in"]),
    # ── Help ──────────────────────────────────────────────────────────
    ("help",          ["help", "what can you do", "list your commands", "what commands",
                       "show commands", "what do you support", "capabilities"]),
    # ── Conversational ────────────────────────────────────────────────
    ("greet",         ["hello", "hi sentinel", "hey sentinel", "good morning",
                       "good afternoon", "good evening", "howdy", "greetings", "what's up"]),
    ("goodbye",       ["bye", "goodbye", "see you", "take care", "i am done",
                       "that is all", "have a good one", "i'm done", "catch you later",
                       "exit sentinel", "quit sentinel", "close sentinel"]),
    ("thanks",        ["thank you", "thanks", "cheers", "appreciate it", "nice one",
                       "brilliant", "great job", "well done", "that is perfect", "excellent",
                       "many thanks"]),
    ("chitchat",      ["how are you", "who are you", "what is your name", "tell me a joke",
                       "are you an ai", "are you sentient", "meaning of life",
                       "talk to me", "say something", "are you smart"]),
    ("clarify",       ["what did you do", "explain that", "what does that mean",
                       "can you repeat", "i do not understand", "what happened",
                       "say that again", "why did you do that"]),
    ("status",        ["what are you doing", "are you busy", "are you ready",
                       "what is your status", "are you working"]),
]


def rule_classify(text: str) -> ClassificationResult:
    """Fast rule-based classifier — used as primary or fallback."""
    t = text.lower().strip()

    # Special case: bare "google X" → web_search  (before open_app picks up "google")
    if re.match(r"^google\s+.+", t):
        return ClassificationResult(
            intent="web_search", confidence=0.90,
            entities={"query": text}, source="rule"
        )

    # Special case: domain-like token in isolation → open_url
    if re.match(r"^(?:go to|visit|open|navigate to)\s+\S+\.\S+", t):
        entities = _extract_entities(text, "open_url")
        return ClassificationResult(
            intent="open_url", confidence=0.88, entities=entities, source="rule"
        )

    for intent, keywords in _RULES:
        for kw in keywords:
            try:
                matched = re.search(kw, t) if any(c in kw for c in r"\.+*?[](){}^$|") else kw in t
            except re.error:
                matched = kw in t
            if matched:
                entities = _extract_entities(text, intent)
                return ClassificationResult(
                    intent=intent,
                    confidence=0.85,
                    entities=entities,
                    source="rule",
                )

    return ClassificationResult(intent="unknown", confidence=0.5, source="rule")


# ─────────────────────────────────────────────
# Main Classifier (neural + rule hybrid)
# ─────────────────────────────────────────────

class SentinelClassifier:
    """
    Intent classifier that uses the trained neural model when available,
    and falls back to rule-based matching otherwise.

    Usage:
        clf = SentinelClassifier()
        clf.load("checkpoints")           # optional — enables neural mode
        result = clf.classify("open chrome")
    """

    def __init__(self):
        # Typed as Optional so Pylance knows they start as None but gain
        # their real types after a successful load().
        from core.llm.model import SentinelTransformer
        from core.llm.tokenizer import SentinelTokenizer
        self.model:         Optional[SentinelTransformer] = None
        self.tokenizer:     Optional[SentinelTokenizer]   = None
        self.intent_labels: List[str] = []
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._neural_ready = False

    def load(self, checkpoint_dir: str = "checkpoints") -> bool:
        """Load neural model. Returns True if successful."""
        try:
            from ..llm.trainer import load_checkpoint
            model, tokenizer, labels = load_checkpoint(checkpoint_dir, str(self.device))
            # Guard: load_checkpoint returns (None, None, None) when no checkpoint exists
            if model is None or tokenizer is None or labels is None:
                return False
            self.model         = model
            self.tokenizer     = tokenizer
            self.intent_labels = labels
            self._neural_ready = True
            print(f"[Classifier] Neural model loaded ({len(labels)} intents, device={self.device})")
            return True
        except Exception as e:
            print(f"[Classifier] Neural load failed: {e} — using rule-based")
            return False

    @torch.no_grad()
    def classify(self, text: str) -> ClassificationResult:
        """Classify text. Returns ClassificationResult."""
        # Try neural first
        if self._neural_ready:
            try:
                result = self._neural_classify(text)
                # If neural is very uncertain, blend with rule
                if result.confidence < 0.55:
                    rule_result = rule_classify(text)
                    if rule_result.intent != "unknown":
                        result.entities.update(rule_result.entities)
                return result
            except Exception as e:
                print(f"[Classifier] Neural error: {e}")

        # Fall back to rules
        return rule_classify(text)

    @torch.no_grad()
    def _neural_classify(self, text: str) -> ClassificationResult:
        # These are guaranteed non-None when _neural_ready is True,
        # but we assert here so Pylance narrows the types correctly.
        assert self.tokenizer is not None
        assert self.model is not None

        ids, mask = self.tokenizer.encode(
            text, max_length=128, padding=True, return_mask=True
        )
        input_ids    = torch.tensor([ids],  dtype=torch.long).to(self.device)
        padding_mask = torch.tensor([mask], dtype=torch.bool).to(self.device)

        logits = self.model.classify(input_ids, padding_mask)      # (1, num_intents)
        probs  = torch.softmax(logits, dim=-1)[0]
        # .item() returns Number; cast to int so it can index lists and tensors
        top_id = int(probs.argmax().item())
        conf   = float(probs[top_id].item())

        intent = (self.intent_labels[top_id]
                  if top_id < len(self.intent_labels)
                  else "unknown")

        entities = _extract_entities(text, intent)

        return ClassificationResult(
            intent=intent,
            confidence=conf,
            entities=entities,
            source="neural",
        )

    @property
    def mode(self) -> str:
        return "neural" if self._neural_ready else "rule-based"