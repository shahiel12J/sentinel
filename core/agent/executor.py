"""
Sentinel Executor

Runs ExecutionPlans produced by the Planner.
Dispatches each ActionStep to the appropriate tool,
reports progress via a callback, and logs results to Memory.
"""

import threading
from typing import Callable, Dict, Any, Optional, List

from .classifier import ClassificationResult
from .memory     import SentinelMemory
from .planner    import ExecutionPlan, ActionStep
from ..tools.apps   import AppLauncher
from ..tools.files  import FileTools
from ..tools.system import SystemTools


# ─────────────────────────────────────────────
# Step status
# ─────────────────────────────────────────────

STEP_PENDING  = "pending"
STEP_RUNNING  = "running"
STEP_DONE     = "done"
STEP_FAILED   = "failed"
STEP_SKIPPED  = "skipped"


# ─────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────

class SentinelExecutor:
    """
    Runs a plan asynchronously and fires callbacks for the UI.

    Callbacks
    ---------
    on_step_start(step_index, description)
    on_step_done(step_index, status, message)
    on_plan_done(success, summary)
    on_output(text)             — for direct text output (system_info, summaries)
    """

    def __init__(self, memory: SentinelMemory):
        self.memory = memory

        # Tool instances
        self.apps   = AppLauncher()
        self.files  = FileTools()
        self.system = SystemTools()

        # Callbacks (set by UI before calling execute)
        self.on_step_start: Optional[Callable] = None
        self.on_step_done:  Optional[Callable] = None
        self.on_plan_done:  Optional[Callable] = None
        self.on_output:     Optional[Callable] = None

        # State shared across steps within a plan
        self._step_context: Dict[str, Any] = {}

    # ── Public API ───────────────────────────────────────────────────

    def execute(self, plan: ExecutionPlan) -> None:
        """Execute a plan in a background thread."""
        self._step_context = {}
        t = threading.Thread(
            target=self._run_plan,
            args=(plan,),
            daemon=True
        )
        t.start()

    # ── Internal ─────────────────────────────────────────────────────

    def _run_plan(self, plan: ExecutionPlan) -> None:
        if not plan.steps:
            # Conversational/soft intents have no steps but are not failures.
            # Only "unknown" is a genuine failure.
            is_success = plan.intent != "unknown"
            self.memory.log_command(
                plan.raw_command, plan.intent,
                success=is_success,
                summary=(plan.preamble or "")[:80],
            )
            self._fire_plan_done(is_success, plan.preamble or "")
            return

        overall_success = True
        summary_parts:  List[str] = []

        for i, step in enumerate(plan.steps):
            self._fire_step_start(i, step.description)

            try:
                success, result = self._dispatch(step, plan)
            except Exception as ex:
                success = False
                result  = f"Unexpected error: {ex}"

            if success:
                self._fire_step_done(i, STEP_DONE, result)
                summary_parts.append(f"✓ {step.description.rstrip(' …')}")
            else:
                self._fire_step_done(i, STEP_FAILED, result)
                summary_parts.append(f"✗ {step.description.rstrip(' …')}: {result}")
                overall_success = False

            # If this step produces textual output (summaries, info, lists)
            if _is_textual_output(plan.intent, step.action):
                self._fire_output(result)

            # If step failed at a critical point, abort
            if not success and step.action in ("find_app",):
                break

        # Log to memory
        self.memory.log_command(
            plan.raw_command,
            plan.intent,
            success=overall_success,
            summary="; ".join(summary_parts[:3])
        )

        self._fire_plan_done(overall_success, "\n".join(summary_parts))

    def _dispatch(self, step: ActionStep, plan: ExecutionPlan) -> tuple:
        """Route a step to the correct tool method and return (success, message)."""
        tool   = step.tool
        action = step.action
        params = dict(step.params)      # copy

        # ── apps ─────────────────────────────────────────────────────
        if tool == "apps":
            if action == "find_app":
                app_key = params.get("app_key") or ""
                # Resolve remembered tokens
                if app_key:
                    resolved = self.memory.resolve_memory_token(app_key)
                    if resolved:
                        app_key = resolved
                path = self.apps.find_app(app_key)
                self._step_context["resolved_path"] = path
                self._step_context["app_key"]       = app_key
                if path:
                    return True, f"Found: {path}"
                return False, f"'{app_key}' not found on this system."

            elif action == "launch_app":
                app_key = self._step_context.get("app_key") or params.get("app_key") or ""
                path    = self._step_context.get("resolved_path")
                return self.apps.launch_app(app_key, path)

            elif action == "close_app":
                return self.apps.close_app(params.get("app_key", ""))

            elif action == "web_search":
                return self.apps.web_search(params.get("query", ""))

            elif action == "open_url":
                url = params.get("url", "")
                # Try as a named site first, then as raw URL
                resolved = self.apps.find_app(url)
                if resolved and resolved != url:
                    return self.apps.launch_app(url, resolved)
                return self.apps.open_url(url)

        # ── files ─────────────────────────────────────────────────────
        elif tool == "files":
            if action == "search_files":
                success, msg, paths = self.files.search_files(
                    extension   = params.get("extension", "*"),
                    time_filter = params.get("time_filter", ""),
                    location    = params.get("location", ""),
                    raw_query   = params.get("raw_query", ""),
                )
                self._step_context["found_files"] = paths
                return success, msg

            elif action == "create_folder":
                return self.files.create_folder(
                    name   = params.get("name", "NewFolder"),
                    parent = params.get("parent", ""),
                )

            elif action == "move_files":
                return self.files.move_files(
                    extension   = params.get("extension", "*"),
                    destination = params.get("destination", ""),
                    source      = params.get("source", ""),
                )

            elif action == "copy_files":
                return self.files.copy_files(
                    extension   = params.get("extension", "*"),
                    destination = params.get("destination", ""),
                )

            elif action == "delete_files":
                return self.files.delete_files(
                    extension = params.get("extension", "*"),
                    location  = params.get("location", ""),
                )

            elif action == "rename_file":
                return self.files.rename_file(params.get("raw_command", ""))

            elif action == "open_file":
                return self.files.open_file(
                    filename    = params.get("filename", ""),
                    raw_command = params.get("raw_command", ""),
                )

            elif action == "read_file":
                return self.files.read_file(
                    filename    = params.get("filename", ""),
                    raw_command = params.get("raw_command", ""),
                )

            elif action == "summarize_text":
                return self.files.summarize_text(params.get("filename", ""))

            elif action == "list_files":
                return self.files.list_files(params.get("location", ""))

        # ── system ────────────────────────────────────────────────────
        elif tool == "system":
            if action == "volume_up":
                return self.system.volume_up()
            elif action == "volume_down":
                return self.system.volume_down()
            elif action == "set_volume":
                return self.system.set_volume(int(params.get("level", 50)))
            elif action == "toggle_mute":
                return self.system.toggle_mute()
            elif action == "media_play_pause":
                return self.system.media_play_pause()
            elif action == "media_next":
                return self.system.media_next()
            elif action == "media_prev":
                return self.system.media_prev()
            elif action == "media_stop":
                return self.system.media_stop()
            elif action == "screenshot":
                return self.system.screenshot()
            elif action == "save_screenshot":
                return self.system.save_screenshot()
            elif action == "shutdown":
                return self.system.shutdown(params.get("minutes", 0))
            elif action == "restart":
                return self.system.restart(params.get("minutes", 0))
            elif action == "sleep":
                return self.system.sleep()
            elif action == "lock_screen":
                return self.system.lock_screen()
            elif action == "cancel_shutdown":
                return self.system.cancel_shutdown()
            elif action == "minimize_all":
                return self.system.minimize_all()
            elif action == "brightness_up":
                return self.system.brightness_up(int(params.get("step", 10)))
            elif action == "brightness_down":
                return self.system.brightness_down(int(params.get("step", 10)))
            elif action == "list_processes":
                return self.system.list_processes()
            elif action == "get_time_date":
                return self.system.get_time_date()
            elif action == "system_info":
                return self.system.system_info()
            elif action == "schedule_reminder":
                cb = self.on_output
                return self.system.schedule_reminder(
                    minutes  = params.get("minutes", 5),
                    message  = params.get("message", ""),
                    callback = cb,
                )
            elif action == "show_help":
                return self.system.show_help()

        # ── memory ────────────────────────────────────────────────────
        elif tool == "memory":
            if action == "remember":
                key   = params.get("key", "")
                value = params.get("value", "")
                cmd   = params.get("raw_command", "")

                # Parse from raw command if entities were empty
                if not key or not value:
                    import re
                    m = re.search(r"my\s+([a-z ]+?)\s+is\s+([a-zA-Z0-9_.:\- /\\]+)", cmd, re.IGNORECASE)
                    if m:
                        key   = m.group(1).strip().lower()
                        value = m.group(2).strip()

                if key and value:
                    self.memory.remember(key, value)
                    self.memory.set_preference(key, value)
                    return True, f"Got it — I'll remember that your {key} is '{value}'."
                return False, "Could not parse what to remember. Try: 'remember my X is Y'"

            elif action == "recall":
                cmd   = params.get("raw_command", "")
                prefs = self.memory.all_preferences()
                facts = self.memory.all_facts()

                if not prefs and not facts:
                    return True, "I haven't stored anything yet. Try: 'remember my IDE is Rider'"

                lines = ["Here's what I know about you:"]
                for f in facts:
                    lines.append(f"  • {f['key']}: {f['value']}")
                for k, v in prefs.items():
                    if not any(f["key"] == k for f in facts):
                        lines.append(f"  • {k}: {v}")
                return True, "\n".join(lines)

        return False, f"Unknown tool/action: {tool}.{action}"

    # ── Callbacks ─────────────────────────────────────────────────────

    def _fire_step_start(self, idx: int, desc: str):
        if self.on_step_start:
            self.on_step_start(idx, desc)

    def _fire_step_done(self, idx: int, status: str, msg: str):
        if self.on_step_done:
            self.on_step_done(idx, status, msg)

    def _fire_plan_done(self, success: bool, summary: str):
        if self.on_plan_done:
            self.on_plan_done(success, summary)

    def _fire_output(self, text: str):
        if self.on_output:
            self.on_output(text)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

_TEXTUAL_INTENTS = {
    "summarize_file", "list_files", "system_info", "recall", "help",
    "file_search", "time_query", "process_list",
}
_TEXTUAL_ACTIONS = {
    "summarize_text", "list_files", "system_info", "recall", "show_help",
    "search_files", "get_time_date", "list_processes",
}


def _is_textual_output(intent: str, action: str) -> bool:
    return intent in _TEXTUAL_INTENTS or action in _TEXTUAL_ACTIONS
