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

# "open / close / launch X"  (fallback — primary extraction uses _find_app_in_text)
_APP_PAT = re.compile(
    r"(?:open|launch|start|close|quit|kill|terminate|exit|run|fire up|bring up)\s+"
    r"(?:up\s+)?(?:my\s+)?([a-z][a-z0-9 ]+?)(?:\s+(?:for me|please|now|up|again|right now))?\s*$",
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

# Filler stripping helpers
_POLITE_PREFIX_PAT = re.compile(
    r"^(?:can\s+you|could\s+you|please|would\s+you|"
    r"i(?:'d)?\s+(?:want|need|like)\s+(?:you\s+)?(?:to\s+)?|"
    r"i\s+want\s+to|i\s+need\s+to|help\s+me|go\s+ahead\s+and)\s+",
    re.IGNORECASE
)
_SUFFIX_PAT = re.compile(
    r"\s+(?:for\s+me|please|now|right\s+now|asap|immediately|quickly|again|up)\s*$",
    re.IGNORECASE
)
_ACTION_VERB_PAT = re.compile(
    r"^(?:open|launch|start|close|quit|kill|terminate|exit|run|"
    r"fire\s+up|bring\s+up|shut\s+down|boot\s+up|load)\s+(?:up\s+)?(?:my\s+)?",
    re.IGNORECASE
)


def _strip_fillers(text: str) -> str:
    """Remove polite prefixes and trailing filler phrases."""
    t = _POLITE_PREFIX_PAT.sub("", text.strip())
    t = _SUFFIX_PAT.sub("", t)
    return t.strip()


def _find_app_in_text(text: str) -> Optional[str]:
    """Find the longest matching app alias in text; return canonical app key."""
    t = text.strip().lower()
    for alias in sorted(APP_ALIASES.keys(), key=len, reverse=True):
        try:
            if re.search(r"\b" + re.escape(alias) + r"\b", t):
                val = APP_ALIASES[alias]
                if not val.startswith("__"):
                    return val
        except re.error:
            if alias in t:
                val = APP_ALIASES[alias]
                if not val.startswith("__"):
                    return val
    return None


def _extract_entities(text: str, intent: str) -> Dict[str, str]:
    t = text.strip()
    entities: Dict[str, str] = {}

    if intent in ("open_app", "close_app"):
        # Primary: strip polite wrappers + action verb, then longest-alias match
        t_clean = _strip_fillers(t)
        verb_free = _ACTION_VERB_PAT.sub("", t_clean).strip()
        app = _find_app_in_text(verb_free)
        # Fallback: regex
        if not app:
            m = _APP_PAT.search(t)
            if m:
                app = resolve_app(m.group(1).strip())
        if app:
            entities["app"] = app

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

# ─────────────────────────────────────────────
# Keyword intent signals
# Format: {intent: (trigger_words, context_words)}
# Score = (trigger_hits × 3) + context_hits
# A match requires at least one trigger word.
# ─────────────────────────────────────────────

_INTENT_KEYWORDS: Dict[str, Tuple[frozenset, frozenset]] = {
    # ── Apps ──────────────────────────────────────────────────────────
    "open_app":       (frozenset({"open","launch","start","run","fire","bring","boot","load"}),
                       frozenset({"app","application","program"})),
    "close_app":      (frozenset({"close","quit","kill","terminate","exit","force"}),
                       frozenset()),
    # ── Files ─────────────────────────────────────────────────────────
    "file_search":    (frozenset({"find","search","locate","where","look","seek"}),
                       frozenset({"file","files","document","documents","folder",
                                  "pdf","mp3","png","jpg","mp4","txt","zip","exe",
                                  "log","csv","xlsx","docx","json","py","gif","jpeg"})),
    "create_folder":  (frozenset({"create","make","mkdir","new","add"}),
                       frozenset({"folder","directory","dir"})),
    "move_files":     (frozenset({"move","relocate","transfer","send","put"}),
                       frozenset({"file","files","folder","into"})),
    "copy_files":     (frozenset({"copy","duplicate","clone","backup","replicate"}),
                       frozenset({"file","files","folder"})),
    "delete_files":   (frozenset({"delete","remove","trash","erase","wipe","clean","purge"}),
                       frozenset({"file","files","folder"})),
    "rename_file":    (frozenset({"rename","relabel"}),
                       frozenset()),
    "open_file":      (frozenset({"open","read","view","show","access"}),
                       frozenset({"file","document","doc","pdf","txt","csv","log","spreadsheet"})),
    "list_files":     (frozenset({"list","ls","show","display","see"}),
                       frozenset({"file","files","folder","directory","contents","inside","in"})),
    "summarize_file": (frozenset({"summarize","summarise","summary","gist","overview","tldr"}),
                       frozenset({"file","document","doc","text"})),
    # ── Volume ────────────────────────────────────────────────────────
    "volume_up":      (frozenset({"louder","crank","amplify"}),
                       frozenset()),
    "volume_down":    (frozenset({"quieter","softer"}),
                       frozenset()),
    "set_volume":     (frozenset({"set","put","change","adjust"}),
                       frozenset({"volume"})),
    "mute":           (frozenset({"mute","unmute","silence"}),
                       frozenset({"sound","audio","volume"})),
    # ── Media ─────────────────────────────────────────────────────────
    "media_play":     (frozenset({"play","resume","unpause","continue"}),
                       frozenset({"music","song","track","audio","media","spotify","podcast","playlist"})),
    "media_pause":    (frozenset({"pause"}),
                       frozenset({"music","song","track","spotify","playback","media","playing"})),
    "media_next":     (frozenset({"next","skip","forward","ahead"}),
                       frozenset({"song","track","music"})),
    "media_prev":     (frozenset({"previous","prev","rewind","back"}),
                       frozenset({"song","track","music"})),
    "media_stop":     (frozenset({"stop","halt"}),
                       frozenset({"music","playback","spotify","playing","song","track","audio"})),
    # ── Screenshot ────────────────────────────────────────────────────
    "screenshot":     (frozenset({"screenshot","screengrab","snap","capture","grab"}),
                       frozenset({"screen","display"})),
    # ── Power / session ───────────────────────────────────────────────
    "shutdown":       (frozenset({"shutdown","poweroff"}),
                       frozenset({"computer","pc","machine","system"})),
    "restart":        (frozenset({"restart","reboot","relaunch"}),
                       frozenset({"computer","pc","machine","system"})),
    "sleep":          (frozenset({"sleep","hibernate","suspend"}),
                       frozenset()),
    "lock_screen":    (frozenset({"lock"}),
                       frozenset({"screen","computer","pc","workstation","session"})),
    "cancel_shutdown":(frozenset({"cancel","abort","undo","prevent"}),
                       frozenset({"shutdown","restart","reboot","shutting"})),
    # ── Brightness / display ──────────────────────────────────────────
    "brightness_up":  (frozenset({"brighter","increase","raise","boost"}),
                       frozenset({"brightness","screen","display","monitor"})),
    "brightness_down":(frozenset({"dimmer","dim","decrease","lower","reduce"}),
                       frozenset({"brightness","screen","display","monitor"})),
    "minimize_all":   (frozenset({"minimize","minimise","hide"}),
                       frozenset({"windows","desktop","all","everything"})),
    # ── System info ───────────────────────────────────────────────────
    "system_info":    (frozenset({"cpu","ram","memory","disk","battery","stats",
                                  "info","status","usage","specs","storage","speed"}),
                       frozenset({"system","pc","computer","machine"})),
    "process_list":   (frozenset({"running","active"}),
                       frozenset({"apps","processes","programs","tasks","applications"})),
    # ── Time ──────────────────────────────────────────────────────────
    "time_query":     (frozenset({"time","date","clock","today"}),
                       frozenset()),
    # ── Reminders ─────────────────────────────────────────────────────
    "schedule_task":  (frozenset({"remind","reminder","timer","alarm","notify","alert","schedule"}),
                       frozenset()),
    # ── Web ───────────────────────────────────────────────────────────
    "web_search":     (frozenset({"google","bing","search","lookup","research"}),
                       frozenset({"online","web","internet"})),
    "open_url":       (frozenset({"go","visit","navigate","browse"}),
                       frozenset({"website","url","link","site","page"})),
    # ── Memory ────────────────────────────────────────────────────────
    "remember":       (frozenset({"remember","store","save","note","memorize","memorise","keep","learn"}),
                       frozenset()),
    "recall":         (frozenset({"recall","retrieve"}),
                       frozenset({"my","memory","stored","saved"})),
    # ── Conversational ────────────────────────────────────────────────
    "help":           (frozenset({"help","commands","capabilities","support","guide","instructions"}),
                       frozenset()),
    "greet":          (frozenset({"hello","hi","hey","howdy","greetings","morning","afternoon","evening"}),
                       frozenset()),
    "goodbye":        (frozenset({"bye","goodbye","farewell","later","cya"}),
                       frozenset()),
    "thanks":         (frozenset({"thanks","thank","cheers","appreciate","brilliant","excellent"}),
                       frozenset()),
    "chitchat":       (frozenset({"joke","funny","sentient","meaning","opinion","feelings","clever"}),
                       frozenset()),
    "clarify":        (frozenset({"explain","clarify","repeat","mean"}),
                       frozenset({"what","that","again","why","did","just"})),
    "status":         (frozenset({"ready","busy","working","doing"}),
                       frozenset({"you","are","sentinel"})),
}


def rule_classify(text: str) -> ClassificationResult:
    """
    Keyword-scoring classifier — matches intent from individual words so
    natural phrasing ("can you please open chrome for me") works without
    needing exact phrases.
    """
    t = text.lower().strip()
    words = set(re.findall(r"\b\w+\b", t))

    # ── High-priority exact patterns ─────────────────────────────────

    # "google X" → web_search
    if re.match(r"^google\b.+", t):
        return ClassificationResult(
            intent="web_search", confidence=0.90,
            entities={"query": text}, source="rule"
        )

    # URL in text → open_url
    if re.search(r"(?:https?://|www\.)\S+|\b\w+\.(com|org|net|io|co|uk|edu|gov)\b", t):
        return ClassificationResult(
            intent="open_url", confidence=0.92,
            entities=_extract_entities(text, "open_url"), source="rule"
        )

    # "show desktop" → minimize_all
    if re.search(r"\bshow\s+(?:the\s+)?desktop\b", t):
        return ClassificationResult(intent="minimize_all", confidence=0.92, entities={}, source="rule")

    # "print screen" / "take a screenshot" → screenshot
    if re.search(r"\bprint\s+(?:the\s+)?screen\b|\btake\s+(?:a\s+)?screenshot\b", t):
        return ClassificationResult(intent="screenshot", confidence=0.92, entities={}, source="rule")

    # "turn off" → shutdown (unless audio context)
    if re.search(r"\b(?:turn|power|switch)\s+(?:it\s+)?off\b", t):
        if not words & {"music","sound","audio","song","track","spotify"}:
            return ClassificationResult(
                intent="shutdown", confidence=0.88,
                entities=_extract_entities(text, "shutdown"), source="rule"
            )

    # "turn up/down (the volume)" → volume direction
    if re.search(r"\bturn\s+(?:(?:the|it|that)\s+)?(?:volume\s+)?up\b", t) or \
       (re.search(r"\bturn\s+up\b", t) and "volume" in words):
        return ClassificationResult(intent="volume_up", confidence=0.90, entities={}, source="rule")
    if re.search(r"\bturn\s+(?:(?:the|it|that)\s+)?(?:volume\s+)?down\b", t) or \
       (re.search(r"\bturn\s+down\b", t) and "volume" in words):
        return ClassificationResult(intent="volume_down", confidence=0.90, entities={}, source="rule")

    # "what time/day/date is it" style
    if re.search(
        r"\b(?:what(?:\'s)?\s+(?:the\s+)?(?:time|date|day|today)"
        r"|current\s+(?:time|date)|(?:time|day|date)\s+is\s+it)\b", t
    ):
        return ClassificationResult(intent="time_query", confidence=0.92, entities={}, source="rule")

    # "what's my X" / "what is my X" → recall
    if re.search(r"\bwhat(?:\'s|s)?\s+(?:is\s+)?my\b", t):
        return ClassificationResult(
            intent="recall", confidence=0.88,
            entities=_extract_entities(text, "recall"), source="rule"
        )

    # "don't forget" → remember
    if re.search(r"\bdon(?:'t|t)\s+(?:forget|lose)\b", t):
        return ClassificationResult(
            intent="remember", confidence=0.85,
            entities=_extract_entities(text, "remember"), source="rule"
        )

    # "shut down" / "shut it down" → shutdown
    if re.search(r"\bshut\s+(?:it\s+|the\s+pc\s+|my\s+pc\s+)?down\b", t):
        return ClassificationResult(
            intent="shutdown", confidence=0.88,
            entities=_extract_entities(text, "shutdown"), source="rule"
        )

    # ── Score intents by keyword presence ────────────────────────────

    scores: Dict[str, int] = {}
    for intent, (triggers, context) in _INTENT_KEYWORDS.items():
        hits = words & triggers
        if hits:
            scores[intent] = len(hits) * 3 + len(words & context)

    # ── Context-based score boosts ────────────────────────────────────

    # "louder" / "quieter" directly → volume direction (no "volume" word needed)
    if words & {"louder","crank","amplify"}:
        scores["volume_up"] = scores.get("volume_up", 0) + 10
    if words & {"quieter","softer"}:
        scores["volume_down"] = scores.get("volume_down", 0) + 10

    # schedule_task: boost when reminder word + time unit present
    if words & {"remind","reminder","alarm","timer","notify","alert","schedule"}:
        if words & {"minutes","minute","hours","hour","seconds","second","mins","hrs"}:
            scores["schedule_task"] = scores.get("schedule_task", 0) + 8

    # "bright" + "how" → system_info; otherwise brightness direction
    if words & {"brightness","bright"}:
        if "how" in words:
            scores["system_info"] = scores.get("system_info", 0) + 8
        elif words & {"up","increase","raise","brighter","more","max","higher"}:
            scores["brightness_up"] = scores.get("brightness_up", 0) + 10
        elif words & {"down","decrease","lower","dimmer","less","min","dim"}:
            scores["brightness_down"] = scores.get("brightness_down", 0) + 10

    # Volume direction via "volume" + directional words
    if "volume" in words:
        if words & {"up","increase","raise","higher","louder","more","crank","boost","max","full"}:
            scores["volume_up"] = scores.get("volume_up", 0) + 10
        elif words & {"down","decrease","lower","reduce","softer","quieter","less","drop","min"}:
            scores["volume_down"] = scores.get("volume_down", 0) + 10
        elif words & {"set","put","change","adjust","make"}:
            scores["set_volume"] = scores.get("set_volume", 0) + 8

    # "open/launch/run + known app" → boost open_app
    if words & {"open","launch","start","run","fire","bring","boot"}:
        verb_free = _ACTION_VERB_PAT.sub("", _strip_fillers(t)).strip()
        if _find_app_in_text(verb_free):
            scores["open_app"] = scores.get("open_app", 0) + 8

    # "close/quit/kill + known app" → boost close_app; suppress shutdown
    if words & {"close","quit","kill","terminate","exit"}:
        verb_free = re.sub(
            r"^(?:close|quit|kill|terminate|exit|force\s+quit)\s+(?:my\s+)?", "", _strip_fillers(t), flags=re.I
        ).strip()
        if _find_app_in_text(verb_free):
            scores["close_app"] = scores.get("close_app", 0) + 8
        scores.pop("shutdown", None)

    # "stop" disambiguation: media vs cancel_shutdown
    if "stop" in words:
        if words & {"music","song","track","spotify","playing","playback","audio","media"}:
            scores["media_stop"] = scores.get("media_stop", 0) + 8
            scores.pop("cancel_shutdown", None)
        elif words & {"shutdown","restart","reboot","shutting"}:
            scores["cancel_shutdown"] = scores.get("cancel_shutdown", 0) + 8
            scores.pop("media_stop", None)

    # "cancel/abort" + shutdown context → cancel_shutdown
    if words & {"cancel","abort","undo"}:
        if words & {"shutdown","restart","reboot","shutting"}:
            scores["cancel_shutdown"] = scores.get("cancel_shutdown", 0) + 10

    # "search/find" + web keywords → web_search; file context → file_search
    if words & {"search","find","look","locate"}:
        if words & {"online","web","internet","google","bing"}:
            scores.pop("file_search", None)
            scores["web_search"] = scores.get("web_search", 0) + 8

    # "list/show" + location words → list_files; suppress file_search
    if words & {"list","ls","show","display"} and words & {"file","files","folder","directory","contents","in","inside"}:
        scores["list_files"] = scores.get("list_files", 0) + 6
        scores.pop("file_search", None)

    # "open" + file extension → open_file over open_app
    if "open_app" in scores and "open_file" in scores:
        if _EXT_PAT.search(t):
            scores.pop("open_app", None)
        else:
            scores.pop("open_file", None)

    # "what" questions
    if words & {"what","whats"}:
        if words & {"running","active"} and words & {"apps","processes","programs","tasks","applications"}:
            scores["process_list"] = scores.get("process_list", 0) + 8
        if words & {"can","could","do","does"} and "you" in words:
            scores["help"] = scores.get("help", 0) + 5
        if words & {"time","date","day","clock"}:
            scores["time_query"] = scores.get("time_query", 0) + 8
        if "my" in words:
            scores["recall"] = scores.get("recall", 0) + 6

    if not scores:
        return ClassificationResult(intent="unknown", confidence=0.5, source="rule")

    best = max(scores, key=lambda i: scores[i])
    entities = _extract_entities(text, best)
    return ClassificationResult(intent=best, confidence=0.85, entities=entities, source="rule")


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