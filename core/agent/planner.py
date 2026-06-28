"""
Sentinel Planner

Converts a ClassificationResult into an ordered list of ActionStep objects.
Each step has:
  - description  — what Sentinel tells the user it is doing
  - tool         — which tool module to call
  - params       — parameters for that tool call

The executor runs these steps sequentially, reporting progress back to the UI.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .classifier import ClassificationResult
from .memory     import SentinelMemory


# ─────────────────────────────────────────────
# Step dataclass
# ─────────────────────────────────────────────

@dataclass
class ActionStep:
    description: str          # Human-readable explanation
    tool:        str          # Tool key: "apps", "files", "system"
    action:      str          # Function name inside tool module
    params:      Dict[str, Any] = field(default_factory=dict)
    confirmable: bool = False  # Whether to ask confirmation before running


@dataclass
class ExecutionPlan:
    intent:      str
    raw_command: str
    steps:       List[ActionStep]
    preamble:    str = ""      # What Sentinel says before executing


# ─────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────

class SentinelPlanner:
    """
    Creates ExecutionPlans from classification results.

    Usage:
        planner = SentinelPlanner(memory)
        plan = planner.plan(result, raw_command="open chrome")
    """

    def __init__(self, memory: SentinelMemory):
        self.memory = memory

    def plan(self, result: ClassificationResult, raw_command: str = "") -> ExecutionPlan:
        """Build an ExecutionPlan for the given classification result."""
        intent   = result.intent
        entities = dict(result.entities)    # copy so we can mutate

        # Resolve memory tokens before planning
        self._resolve_memory_tokens(entities)

        method = getattr(self, f"_plan_{intent}", self._plan_unknown)
        return method(entities, raw_command)

    # ── Memory token resolver ────────────────────────────────────────

    def _resolve_memory_tokens(self, entities: Dict[str, Any]) -> None:
        for k, v in list(entities.items()):
            if isinstance(v, str) and v.startswith("__remembered_"):
                resolved = self.memory.resolve_memory_token(v)
                if resolved:
                    entities[k] = resolved
                else:
                    entities[k] = None

    # ── Intent planners ──────────────────────────────────────────────

    def _plan_open_app(self, e: Dict, cmd: str) -> ExecutionPlan:
        app = e.get("app") or "unknown"
        return ExecutionPlan(
            intent="open_app",
            raw_command=cmd,
            preamble=f"Opening {_friendly(app)} for you.",
            steps=[
                ActionStep(
                    description=f"Locating {_friendly(app)} …",
                    tool="apps", action="find_app",
                    params={"app_key": app}
                ),
                ActionStep(
                    description=f"Launching {_friendly(app)} …",
                    tool="apps", action="launch_app",
                    params={"app_key": app}
                ),
            ]
        )

    def _plan_close_app(self, e: Dict, cmd: str) -> ExecutionPlan:
        app = e.get("app") or "unknown"
        return ExecutionPlan(
            intent="close_app",
            raw_command=cmd,
            preamble=f"Closing {_friendly(app)}.",
            steps=[
                ActionStep(
                    description=f"Closing {_friendly(app)} …",
                    tool="apps", action="close_app",
                    params={"app_key": app}
                ),
            ]
        )

    def _plan_file_search(self, e: Dict, cmd: str) -> ExecutionPlan:
        ext      = e.get("extension", "*")
        time_f   = e.get("time_filter", "")
        location = e.get("location", "")
        filename = e.get("filename", "")

        if filename:
            desc = f'Searching for "{filename}"'
        else:
            desc = f"Searching for {'.' + ext if ext != '*' else 'all'} files"
        if time_f == "today":
            desc += " modified today"
        if location:
            desc += f" in {location}"

        return ExecutionPlan(
            intent="file_search",
            raw_command=cmd,
            preamble=f"{desc}.",
            steps=[
                ActionStep(
                    description=desc + " …",
                    tool="files", action="search_files",
                    params={
                        "extension":  ext,
                        "time_filter": time_f,
                        "location":   location,
                        "raw_query":  cmd,
                        "name_query": filename,
                    }
                ),
            ]
        )

    def _plan_create_folder(self, e: Dict, cmd: str) -> ExecutionPlan:
        name = e.get("name", "NewFolder")
        return ExecutionPlan(
            intent="create_folder",
            raw_command=cmd,
            preamble=f"Creating folder '{name}'.",
            steps=[
                ActionStep(
                    description=f"Creating folder '{name}' …",
                    tool="files", action="create_folder",
                    params={"name": name}
                ),
            ]
        )

    def _plan_move_files(self, e: Dict, cmd: str) -> ExecutionPlan:
        ext  = e.get("extension", "*")
        dest = e.get("destination", "")
        label = f".{ext} files" if ext != "*" else "files"
        return ExecutionPlan(
            intent="move_files",
            raw_command=cmd,
            preamble=f"Moving {label} to '{dest}'.",
            steps=[
                ActionStep(
                    description=f"Finding {label} …",
                    tool="files", action="search_files",
                    params={"extension": ext, "location": ""}
                ),
                ActionStep(
                    description=f"Moving {label} to '{dest}' …",
                    tool="files", action="move_files",
                    params={"extension": ext, "destination": dest},
                    confirmable=True
                ),
            ]
        )

    def _plan_copy_files(self, e: Dict, cmd: str) -> ExecutionPlan:
        ext  = e.get("extension", "*")
        dest = e.get("destination", "")
        label = f".{ext} files" if ext != "*" else "files"
        return ExecutionPlan(
            intent="copy_files",
            raw_command=cmd,
            preamble=f"Copying {label} to '{dest}'.",
            steps=[
                ActionStep(
                    description=f"Copying {label} to '{dest}' …",
                    tool="files", action="copy_files",
                    params={"extension": ext, "destination": dest},
                    confirmable=True
                ),
            ]
        )

    def _plan_delete_files(self, e: Dict, cmd: str) -> ExecutionPlan:
        ext = e.get("extension", "tmp")
        return ExecutionPlan(
            intent="delete_files",
            raw_command=cmd,
            preamble=f"Preparing to delete .{ext} files.",
            steps=[
                ActionStep(
                    description=f"Finding .{ext} files …",
                    tool="files", action="search_files",
                    params={"extension": ext}
                ),
                ActionStep(
                    description=f"Deleting .{ext} files …",
                    tool="files", action="delete_files",
                    params={"extension": ext},
                    confirmable=True
                ),
            ]
        )

    def _plan_rename_file(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent="rename_file",
            raw_command=cmd,
            preamble="Renaming file.",
            steps=[
                ActionStep(
                    description="Renaming file …",
                    tool="files", action="rename_file",
                    params={"raw_command": cmd}
                ),
            ]
        )

    def _plan_open_file(self, e: Dict, cmd: str) -> ExecutionPlan:
        fname = e.get("filename", "")
        return ExecutionPlan(
            intent="open_file",
            raw_command=cmd,
            preamble=f"Opening {fname or 'file'}.",
            steps=[
                ActionStep(
                    description=f"Opening {fname or 'file'} …",
                    tool="files", action="open_file",
                    params={"filename": fname, "raw_command": cmd}
                ),
            ]
        )

    def _plan_summarize_file(self, e: Dict, cmd: str) -> ExecutionPlan:
        fname = e.get("filename", "")
        return ExecutionPlan(
            intent="summarize_file",
            raw_command=cmd,
            preamble=f"Reading and summarizing {fname or 'file'} …",
            steps=[
                ActionStep(
                    description="Loading file content …",
                    tool="files", action="read_file",
                    params={"filename": fname, "raw_command": cmd}
                ),
                ActionStep(
                    description="Generating summary …",
                    tool="files", action="summarize_text",
                    params={"filename": fname}
                ),
            ]
        )

    def _plan_list_files(self, e: Dict, cmd: str) -> ExecutionPlan:
        location = e.get("location", "")
        return ExecutionPlan(
            intent="list_files",
            raw_command=cmd,
            preamble=f"Listing files{' in ' + location if location else ''}.",
            steps=[
                ActionStep(
                    description=f"Listing files{' in ' + location if location else ''} …",
                    tool="files", action="list_files",
                    params={"location": location}
                ),
            ]
        )

    def _plan_remember(self, e: Dict, cmd: str) -> ExecutionPlan:
        key   = e.get("key", "")
        value = e.get("value", "")
        return ExecutionPlan(
            intent="remember",
            raw_command=cmd,
            preamble=f"Storing: {key} = {value}",
            steps=[
                ActionStep(
                    description=f"Saving '{key}' to memory …",
                    tool="memory", action="remember",
                    params={"key": key, "value": value, "raw_command": cmd}
                ),
            ]
        )

    def _plan_recall(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent="recall",
            raw_command=cmd,
            preamble="Searching memory …",
            steps=[
                ActionStep(
                    description="Looking up stored information …",
                    tool="memory", action="recall",
                    params={"raw_command": cmd}
                ),
            ]
        )

    def _plan_volume_up(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent="volume_up", raw_command=cmd, preamble="Increasing volume.",
            steps=[ActionStep(description="Adjusting volume up …",
                              tool="system", action="volume_up", params={})]
        )

    def _plan_volume_down(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent="volume_down", raw_command=cmd, preamble="Decreasing volume.",
            steps=[ActionStep(description="Adjusting volume down …",
                              tool="system", action="volume_down", params={})]
        )

    def _plan_mute(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent="mute", raw_command=cmd, preamble="Toggling mute.",
            steps=[ActionStep(description="Toggling mute …",
                              tool="system", action="toggle_mute", params={})]
        )

    def _plan_screenshot(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent="screenshot", raw_command=cmd, preamble="Taking a screenshot.",
            steps=[
                ActionStep(description="Capturing screen …",
                           tool="system", action="screenshot", params={}),
                ActionStep(description="Saving screenshot …",
                           tool="system", action="save_screenshot", params={}),
            ]
        )

    def _plan_shutdown(self, e: Dict, cmd: str) -> ExecutionPlan:
        mins = int(e.get("minutes", 0))
        label = f"in {mins} minute{'s' if mins != 1 else ''}" if mins else "now"
        return ExecutionPlan(
            intent="shutdown", raw_command=cmd,
            preamble=f"Scheduling shutdown {label}.",
            steps=[
                ActionStep(
                    description=f"Scheduling shutdown {label} …",
                    tool="system", action="shutdown",
                    params={"minutes": mins},
                    confirmable=True
                ),
            ]
        )

    def _plan_restart(self, e: Dict, cmd: str) -> ExecutionPlan:
        mins = int(e.get("minutes", 0))
        return ExecutionPlan(
            intent="restart", raw_command=cmd,
            preamble="Scheduling restart.",
            steps=[
                ActionStep(
                    description="Scheduling restart …",
                    tool="system", action="restart",
                    params={"minutes": mins},
                    confirmable=True
                ),
            ]
        )

    def _plan_sleep(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent="sleep", raw_command=cmd, preamble="Putting computer to sleep.",
            steps=[
                ActionStep(description="Initiating sleep …",
                           tool="system", action="sleep",
                           params={}, confirmable=True)
            ]
        )

    def _plan_system_info(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent="system_info", raw_command=cmd, preamble="Gathering system information.",
            steps=[
                ActionStep(description="Reading system stats …",
                           tool="system", action="system_info", params={})
            ]
        )

    def _plan_schedule_task(self, e: Dict, cmd: str) -> ExecutionPlan:
        mins = int(e.get("minutes", 0))
        return ExecutionPlan(
            intent="schedule_task", raw_command=cmd,
            preamble=f"Setting reminder in {mins} minute{'s' if mins != 1 else ''}.",
            steps=[
                ActionStep(
                    description=f"Scheduling reminder in {mins} min …",
                    tool="system", action="schedule_reminder",
                    params={"minutes": mins, "message": cmd}
                )
            ]
        )

    def _plan_set_volume(self, e: Dict, cmd: str) -> ExecutionPlan:
        level = int(e.get("level", 50))
        return ExecutionPlan(
            intent="set_volume", raw_command=cmd,
            preamble=f"Setting volume to {level}%.",
            steps=[ActionStep(description=f"Setting volume to {level}% …",
                              tool="system", action="set_volume",
                              params={"level": level})]
        )

    def _plan_media_play(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(intent="media_play", raw_command=cmd,
            preamble="Playing media.",
            steps=[ActionStep(description="Sending play …",
                              tool="system", action="media_play_pause", params={})])

    def _plan_media_pause(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(intent="media_pause", raw_command=cmd,
            preamble="Pausing media.",
            steps=[ActionStep(description="Sending pause …",
                              tool="system", action="media_play_pause", params={})])

    def _plan_media_next(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(intent="media_next", raw_command=cmd,
            preamble="Skipping to next track.",
            steps=[ActionStep(description="Next track …",
                              tool="system", action="media_next", params={})])

    def _plan_media_prev(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(intent="media_prev", raw_command=cmd,
            preamble="Going to previous track.",
            steps=[ActionStep(description="Previous track …",
                              tool="system", action="media_prev", params={})])

    def _plan_media_stop(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(intent="media_stop", raw_command=cmd,
            preamble="Stopping media.",
            steps=[ActionStep(description="Stopping playback …",
                              tool="system", action="media_stop", params={})])

    def _plan_lock_screen(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(intent="lock_screen", raw_command=cmd,
            preamble="Locking screen.",
            steps=[ActionStep(description="Locking screen …",
                              tool="system", action="lock_screen", params={})])

    def _plan_cancel_shutdown(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(intent="cancel_shutdown", raw_command=cmd,
            preamble="Cancelling scheduled shutdown.",
            steps=[ActionStep(description="Cancelling shutdown …",
                              tool="system", action="cancel_shutdown", params={})])

    def _plan_brightness_up(self, e: Dict, cmd: str) -> ExecutionPlan:
        step = int(e.get("step", 10))
        return ExecutionPlan(intent="brightness_up", raw_command=cmd,
            preamble="Increasing brightness.",
            steps=[ActionStep(description="Adjusting brightness up …",
                              tool="system", action="brightness_up",
                              params={"step": step})])

    def _plan_brightness_down(self, e: Dict, cmd: str) -> ExecutionPlan:
        step = int(e.get("step", 10))
        return ExecutionPlan(intent="brightness_down", raw_command=cmd,
            preamble="Decreasing brightness.",
            steps=[ActionStep(description="Adjusting brightness down …",
                              tool="system", action="brightness_down",
                              params={"step": step})])

    def _plan_minimize_all(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(intent="minimize_all", raw_command=cmd,
            preamble="Minimizing all windows.",
            steps=[ActionStep(description="Showing desktop …",
                              tool="system", action="minimize_all", params={})])

    def _plan_process_list(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(intent="process_list", raw_command=cmd,
            preamble="Listing running processes.",
            steps=[ActionStep(description="Reading running processes …",
                              tool="system", action="list_processes", params={})])

    def _plan_time_query(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(intent="time_query", raw_command=cmd,
            preamble="Checking the clock.",
            steps=[ActionStep(description="Getting current time and date …",
                              tool="system", action="get_time_date", params={})])

    def _plan_open_url(self, e: Dict, cmd: str) -> ExecutionPlan:
        url = e.get("url", cmd)
        return ExecutionPlan(intent="open_url", raw_command=cmd,
            preamble=f"Opening {url} in your browser.",
            steps=[ActionStep(description=f"Opening {url} …",
                              tool="apps", action="open_url",
                              params={"url": url})])

    def _plan_web_search(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent="web_search", raw_command=cmd,
            preamble="Opening browser to search.",
            steps=[
                ActionStep(
                    description="Opening search in browser …",
                    tool="apps", action="web_search",
                    params={"query": cmd}
                )
            ]
        )

    def _plan_help(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent="help", raw_command=cmd,
            preamble="Here's everything I can do:",
            steps=[
                ActionStep(description="Loading command reference …",
                           tool="system", action="show_help", params={})
            ]
        )

    def _plan_unknown(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent="unknown", raw_command=cmd,
            preamble="I'm not sure how to handle that. Try asking for 'help'.",
            steps=[]
        )

    def _plan_greet(self, e: Dict, cmd: str) -> ExecutionPlan:
        import random
        return ExecutionPlan(
            intent="greet", raw_command=cmd,
            preamble=random.choice([
                "Hey! What can I do for you?",
                "Hello! Ready when you are.",
                "Hi there! What do you need?",
                "Hey! How can I help?",
            ]),
            steps=[]
        )

    def _plan_goodbye(self, e: Dict, cmd: str) -> ExecutionPlan:
        import random
        return ExecutionPlan(
            intent="goodbye", raw_command=cmd,
            preamble=random.choice([
                "See you later! Take care.",
                "Goodbye! Come back whenever you need me.",
                "Later! Stay productive.",
                "Bye! Sentinel standing by.",
            ]),
            steps=[]
        )

    def _plan_thanks(self, e: Dict, cmd: str) -> ExecutionPlan:
        import random
        return ExecutionPlan(
            intent="thanks", raw_command=cmd,
            preamble=random.choice([
                "No problem at all!",
                "Happy to help!",
                "Anytime.",
                "Glad that worked!",
            ]),
            steps=[]
        )

    def _plan_chitchat(self, e: Dict, cmd: str) -> ExecutionPlan:
        import random
        t = cmd.lower()
        if any(w in t for w in ["name", "who are you", "what are you"]):
            reply = "I'm Sentinel, your local desktop AI assistant."
        elif any(w in t for w in ["how are you", "doing", "feeling"]):
            reply = "Running at full capacity! All systems nominal."
        elif any(w in t for w in ["joke", "funny"]):
            reply = random.choice([
                "Why do programmers prefer dark mode? Because light attracts bugs.",
                "I told my RAM a joke. It forgot it immediately.",
            ])
        elif "meaning of life" in t:
            reply = "42. Or possibly 'open more Chrome tabs' — the data is unclear."
        else:
            reply = random.choice([
                "I'm better at opening apps than deep conversation, but I'm listening.",
                "Interesting thought. Need me to do anything while we chat?",
            ])
        return ExecutionPlan(intent="chitchat", raw_command=cmd, preamble=reply, steps=[])

    def _plan_clarify(self, e: Dict, cmd: str) -> ExecutionPlan:
        history = self.memory.recent_history(1)
        if history:
            last = history[0]
            reply = (
                f"My last action was: '{last.get('command', '?')}' "
                f"(intent: {last.get('intent', '?')}, "
                f"result: {'succeeded' if last.get('success') else 'failed'})."
            )
        else:
            reply = "I haven't done anything yet this session. Give me a command!"
        return ExecutionPlan(intent="clarify", raw_command=cmd, preamble=reply, steps=[])

    def _plan_status(self, e: Dict, cmd: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent="status", raw_command=cmd,
            preamble="I'm idle and ready for your next command.",
            steps=[]
        )


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

_FRIENDLY: Dict[str, str] = {
    "chrome":        "Google Chrome",
    "visual_studio": "Visual Studio",
    "vscode":        "VS Code",
    "rider":         "JetBrains Rider",
    "github_desktop":"GitHub Desktop",
    "outlook":       "Microsoft Outlook",
    "task_manager":  "Task Manager",
    "notepad":       "Notepad",
    "mspaint":       "Paint",
    "calculator":    "Calculator",
    "explorer":      "File Explorer",
    "winword":       "Microsoft Word",
    "excel":         "Microsoft Excel",
    "spotify":       "Spotify",
    "discord":       "Discord",
    "slack":         "Slack",
    "steam":         "Steam",
    "teams":         "Microsoft Teams",
    "cmd":           "Command Prompt",
    "powershell":    "PowerShell",
}


def _friendly(app_key: str) -> str:
    return _FRIENDLY.get(app_key, app_key.replace("_", " ").title())


