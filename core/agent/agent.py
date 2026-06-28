"""
sentinel/core/agent/agent.py
────────────────────────────
Central agent facade.  The UI calls:

    agent.process_command(
        text,
        on_preamble   = fn(str),
        on_step_start = fn(index, description),
        on_step_done  = fn(index, status, message),
        on_plan_done  = fn(success, summary),
        on_output     = fn(text),
    )

All callbacks are fired from the executor's background thread; the UI
must bounce them to the main thread via Qt signals (already done in
AgentWorker).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from core.agent.classifier import SentinelClassifier
from core.agent.planner    import SentinelPlanner
from core.agent.executor   import SentinelExecutor
from core.agent.memory     import SentinelMemory

log = logging.getLogger(__name__)


class SentinelAgent:
    """
    High-level orchestrator.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory that contains ``sentinel_best.pt`` and ``tokenizer.json``
        produced by the trainer.  May not exist yet — the agent falls back
        to rule-based classification silently.
    db_path : str | Path
        Path to the SQLite database file.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        db_path: str | Path = "data/sentinel.db",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.db_path = Path(db_path)

        log.info("Initialising Sentinel Agent…")

        # Memory (always available)
        self.memory = SentinelMemory(db_path=str(self.db_path))

        # Classifier — tries to load neural checkpoint; falls back to rules
        self.classifier = SentinelClassifier()
        self.classifier.load(str(self.checkpoint_dir))
        mode = self.classifier.mode
        log.info("Classifier ready  (%s)", mode)

        # Planner
        self.planner = SentinelPlanner(memory=self.memory)

        # Executor
        self.executor = SentinelExecutor(memory=self.memory)

        log.info("Sentinel Agent ready.")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def classifier_mode(self) -> str:
        return self.classifier.mode

    def process_command(
        self,
        command: str,
        on_preamble:   Optional[Callable[[str], None]] = None,
        on_step_start: Optional[Callable[[int, str], None]] = None,
        on_step_done:  Optional[Callable[[int, str, str], None]] = None,
        on_plan_done:  Optional[Callable[[bool, str], None]] = None,
        on_output:     Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Full pipeline: classify → plan → execute.

        This method is designed to be called from a background thread
        (AgentWorker).  It returns as soon as execution is complete.
        """
        text = command.strip()
        if not text:
            if on_plan_done:
                on_plan_done(False, "No command provided.")
            return

        # ── 1. Classify ────────────────────────────────────────────────
        try:
            result = self.classifier.classify(text)
            log.info(
                "Classified '%s' → intent=%s  conf=%.2f  src=%s",
                text, result.intent, result.confidence, result.source,
            )
        except Exception as exc:
            log.exception("Classification failed")
            if on_plan_done:
                on_plan_done(False, f"Classification error: {exc}")
            return

        # ── 2. Plan ────────────────────────────────────────────────────
        try:
            plan = self.planner.plan(result, text)
            log.info(
                "Plan ready for '%s': %d step(s)", result.intent, len(plan.steps)
            )
        except Exception as exc:
            log.exception("Planning failed")
            if on_plan_done:
                on_plan_done(False, f"Planning error: {exc}")
            return

        # ── 3. Preamble callback ────────────────────────────────────────
        if on_preamble and plan.preamble:
            try:
                on_preamble(plan.preamble)
            except Exception:
                log.exception("on_preamble callback raised")

        # ── 4. Wire executor callbacks ─────────────────────────────────
        self.executor.on_step_start = on_step_start
        self.executor.on_step_done  = on_step_done
        self.executor.on_plan_done  = on_plan_done
        self.executor.on_output     = on_output

        # ── 5. Execute ─────────────────────────────────────────────────
        try:
            self.executor.execute(plan)
        except Exception as exc:
            log.exception("Execution failed")
            if on_plan_done:
                on_plan_done(False, f"Execution error: {exc}")

    # ------------------------------------------------------------------ #
    #  Memory convenience pass-throughs                                   #
    # ------------------------------------------------------------------ #

    def get_recent_history(self, n: int = 10) -> list[dict]:
        return self.memory.recent_history(n)

    def get_preference(self, key: str) -> Optional[str]:
        return self.memory.get_preference(key)

    def recall(self, query: str) -> Optional[str]:
        return self.memory.fuzzy_recall(query)