"""
data/training_db.py
────────────────────
SQLite-backed training data store.

Tables
------
  training_examples   (id, text, intent, source, created_at)
  intents             (name, description, created_at)

Usage
-----
  from data.training_db import TrainingDB
  db = TrainingDB("data/training.db")
  db.seed_defaults()                          # first-run population
  examples = db.get_all()                     # → [{"text":..,"intent":..}]
  labels   = db.get_intent_labels()           # → ["close_app", ...]
  db.add("hey sentinel", "greet")             # add one example
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Default intent metadata
# ──────────────────────────────────────────────────────────────────────────────

INTENT_META: Dict[str, str] = {
    # ── System control ────────────────────────────────────────────────
    "open_app":       "Launch a desktop application",
    "close_app":      "Close / quit a running application",
    "file_search":    "Search for files by name, type, or date",
    "create_folder":  "Create a new directory",
    "move_files":     "Move files from one location to another",
    "copy_files":     "Copy files to a destination",
    "delete_files":   "Delete files or folders",
    "rename_file":    "Rename a file or folder",
    "open_file":      "Open a file with its default app",
    "summarize_file": "Summarise the contents of a text file",
    "list_files":     "List the contents of a directory",
    "remember":       "Store a user preference or fact",
    "recall":         "Retrieve a stored preference or fact",
    "volume_up":      "Increase system volume",
    "volume_down":    "Decrease system volume",
    "mute":           "Mute or unmute system audio",
    "screenshot":     "Capture a screenshot",
    "shutdown":       "Shut down the computer",
    "restart":        "Restart the computer",
    "sleep":          "Put the computer to sleep",
    "system_info":    "Display CPU, RAM, disk, and network stats",
    "schedule_task":  "Schedule a reminder or timed action",
    "web_search":     "Search the web via the default browser",
    "help":           "Show what Sentinel can do",
    "unknown":        "Command not recognised",
    # ── Conversation ─────────────────────────────────────────────────
    "greet":          "User greeting or salutation",
    "goodbye":        "User farewell or exit signal",
    "thanks":         "User expressing gratitude",
    "chitchat":       "General small talk or casual conversation",
    "clarify":        "User asking Sentinel to explain or repeat itself",
    "status":         "User asking what Sentinel is doing right now",
}


# ──────────────────────────────────────────────────────────────────────────────
# Default training examples
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_EXAMPLES: List[Dict] = [

    # ── open_app ──────────────────────────────────────────────────────
    {"text": "open chrome",                        "intent": "open_app"},
    {"text": "launch chrome",                      "intent": "open_app"},
    {"text": "start chrome",                       "intent": "open_app"},
    {"text": "open google chrome",                 "intent": "open_app"},
    {"text": "open visual studio",                 "intent": "open_app"},
    {"text": "launch visual studio",               "intent": "open_app"},
    {"text": "start visual studio",                "intent": "open_app"},
    {"text": "open vs",                            "intent": "open_app"},
    {"text": "open visual studio code",            "intent": "open_app"},
    {"text": "open vscode",                        "intent": "open_app"},
    {"text": "launch vscode",                      "intent": "open_app"},
    {"text": "open code editor",                   "intent": "open_app"},
    {"text": "open rider",                         "intent": "open_app"},
    {"text": "launch rider",                       "intent": "open_app"},
    {"text": "start jetbrains rider",              "intent": "open_app"},
    {"text": "open github desktop",                "intent": "open_app"},
    {"text": "launch github desktop",              "intent": "open_app"},
    {"text": "open github",                        "intent": "open_app"},
    {"text": "open outlook",                       "intent": "open_app"},
    {"text": "launch outlook",                     "intent": "open_app"},
    {"text": "open my email",                      "intent": "open_app"},
    {"text": "open notepad",                       "intent": "open_app"},
    {"text": "launch task manager",                "intent": "open_app"},
    {"text": "open task manager",                  "intent": "open_app"},
    {"text": "open file explorer",                 "intent": "open_app"},
    {"text": "open calculator",                    "intent": "open_app"},
    {"text": "start the browser",                  "intent": "open_app"},
    {"text": "open my ide",                        "intent": "open_app"},
    {"text": "launch my editor",                   "intent": "open_app"},
    {"text": "open paint",                         "intent": "open_app"},
    {"text": "open spotify",                       "intent": "open_app"},
    {"text": "launch discord",                     "intent": "open_app"},
    {"text": "open teams",                         "intent": "open_app"},
    {"text": "open microsoft teams",               "intent": "open_app"},
    {"text": "start word",                         "intent": "open_app"},
    {"text": "open excel",                         "intent": "open_app"},
    {"text": "launch powerpoint",                  "intent": "open_app"},
    {"text": "open slack",                         "intent": "open_app"},
    {"text": "can you open chrome for me",         "intent": "open_app"},
    {"text": "fire up visual studio",              "intent": "open_app"},
    {"text": "bring up rider",                     "intent": "open_app"},
    {"text": "i need chrome",                      "intent": "open_app"},
    {"text": "get outlook going",                  "intent": "open_app"},
    {"text": "boot up vscode",                     "intent": "open_app"},
    {"text": "pull up file explorer",              "intent": "open_app"},
    {"text": "run notepad",                        "intent": "open_app"},
    {"text": "start spotify",                      "intent": "open_app"},

    # ── close_app ─────────────────────────────────────────────────────
    {"text": "close chrome",                       "intent": "close_app"},
    {"text": "quit chrome",                        "intent": "close_app"},
    {"text": "kill chrome",                        "intent": "close_app"},
    {"text": "close visual studio",                "intent": "close_app"},
    {"text": "quit outlook",                       "intent": "close_app"},
    {"text": "close notepad",                      "intent": "close_app"},
    {"text": "terminate rider",                    "intent": "close_app"},
    {"text": "shut down vscode",                   "intent": "close_app"},
    {"text": "exit file explorer",                 "intent": "close_app"},

    # ── file_search ───────────────────────────────────────────────────
    {"text": "find all pdfs",                      "intent": "file_search"},
    {"text": "search for pdf files",               "intent": "file_search"},
    {"text": "find every pdf modified today",      "intent": "file_search"},
    {"text": "look for pdfs on the desktop",       "intent": "file_search"},
    {"text": "search for word documents",          "intent": "file_search"},
    {"text": "find all docx files",                "intent": "file_search"},
    {"text": "find images from this week",         "intent": "file_search"},
    {"text": "search for png files",               "intent": "file_search"},
    {"text": "find all text files on desktop",     "intent": "file_search"},
    {"text": "look for excel files",               "intent": "file_search"},
    {"text": "find all spreadsheets",              "intent": "file_search"},
    {"text": "search for mp4 files",               "intent": "file_search"},
    {"text": "find videos on my desktop",          "intent": "file_search"},
    {"text": "find all zip files",                 "intent": "file_search"},
    {"text": "look for python files",              "intent": "file_search"},
    {"text": "find py files in documents",         "intent": "file_search"},
    {"text": "search for log files",               "intent": "file_search"},
    {"text": "find files from today",              "intent": "file_search"},
    {"text": "search for recently modified files", "intent": "file_search"},

    # ── create_folder ─────────────────────────────────────────────────
    {"text": "create a folder called projects",    "intent": "create_folder"},
    {"text": "make a new folder named backup",     "intent": "create_folder"},
    {"text": "create directory work",              "intent": "create_folder"},
    {"text": "new folder on desktop called demo",  "intent": "create_folder"},
    {"text": "make folder temp",                   "intent": "create_folder"},
    {"text": "create a folder archive",            "intent": "create_folder"},
    {"text": "mkdir data",                         "intent": "create_folder"},
    {"text": "make a new directory called output", "intent": "create_folder"},
    {"text": "create folder named assets",         "intent": "create_folder"},
    {"text": "create a new folder here",           "intent": "create_folder"},

    # ── move_files ────────────────────────────────────────────────────
    {"text": "move all pdfs to documents",         "intent": "move_files"},
    {"text": "move the images into pictures",      "intent": "move_files"},
    {"text": "move files from desktop to downloads","intent": "move_files"},
    {"text": "relocate the docx files",            "intent": "move_files"},
    {"text": "move everything to backup folder",   "intent": "move_files"},
    {"text": "transfer the pdfs to archive",       "intent": "move_files"},
    {"text": "move these files to the project folder","intent": "move_files"},
    {"text": "shift the images to pictures",       "intent": "move_files"},
    {"text": "move all logs to archive",           "intent": "move_files"},
    {"text": "put the pdfs in documents",          "intent": "move_files"},

    # ── copy_files ────────────────────────────────────────────────────
    {"text": "copy all pdfs to documents",         "intent": "copy_files"},
    {"text": "duplicate the images into backup",   "intent": "copy_files"},
    {"text": "copy the report to desktop",         "intent": "copy_files"},
    {"text": "make a copy of the files",           "intent": "copy_files"},
    {"text": "copy everything to backup",          "intent": "copy_files"},

    # ── delete_files ──────────────────────────────────────────────────
    {"text": "delete all temp files",              "intent": "delete_files"},
    {"text": "remove the old logs",                "intent": "delete_files"},
    {"text": "delete the backup folder",           "intent": "delete_files"},
    {"text": "trash all pdfs on desktop",          "intent": "delete_files"},
    {"text": "erase the temp directory",           "intent": "delete_files"},

    # ── rename_file ───────────────────────────────────────────────────
    {"text": "rename report to final report",      "intent": "rename_file"},
    {"text": "rename the file to backup",          "intent": "rename_file"},
    {"text": "rename old log to archive log",      "intent": "rename_file"},

    # ── open_file ─────────────────────────────────────────────────────
    {"text": "open the readme file",               "intent": "open_file"},
    {"text": "open report.docx",                   "intent": "open_file"},
    {"text": "open that pdf",                      "intent": "open_file"},
    {"text": "open the spreadsheet",               "intent": "open_file"},
    {"text": "open config.json",                   "intent": "open_file"},
    {"text": "open the log file",                  "intent": "open_file"},

    # ── summarize_file ────────────────────────────────────────────────
    {"text": "summarize the readme",               "intent": "summarize_file"},
    {"text": "give me a summary of report.txt",    "intent": "summarize_file"},
    {"text": "what does this file say",            "intent": "summarize_file"},
    {"text": "summarize the document",             "intent": "summarize_file"},
    {"text": "tldr this file",                     "intent": "summarize_file"},
    {"text": "give me a summary of that text file","intent": "summarize_file"},
    {"text": "read and summarize notes.txt",       "intent": "summarize_file"},
    {"text": "what is in this document",           "intent": "summarize_file"},
    {"text": "briefly summarize the log file",     "intent": "summarize_file"},

    # ── list_files ────────────────────────────────────────────────────
    {"text": "list files on desktop",              "intent": "list_files"},
    {"text": "show me what is in downloads",       "intent": "list_files"},
    {"text": "what is in the documents folder",    "intent": "list_files"},
    {"text": "show files in current directory",    "intent": "list_files"},
    {"text": "list everything on the desktop",     "intent": "list_files"},
    {"text": "show folder contents",               "intent": "list_files"},

    # ── remember ──────────────────────────────────────────────────────
    {"text": "remember that my ide is rider",      "intent": "remember"},
    {"text": "save that my project folder is c drive projects","intent": "remember"},
    {"text": "remember my name is alex",           "intent": "remember"},
    {"text": "store that my preferred browser is chrome","intent": "remember"},
    {"text": "remember that i use dark mode",      "intent": "remember"},
    {"text": "note that my backup drive is d",     "intent": "remember"},
    {"text": "save my editor preference as vscode","intent": "remember"},
    {"text": "remember i prefer light theme",      "intent": "remember"},
    {"text": "store my name",                      "intent": "remember"},
    {"text": "keep note that my shell is powershell","intent": "remember"},
    {"text": "remember that my favourite ide is rider","intent": "remember"},

    # ── recall ────────────────────────────────────────────────────────
    {"text": "what is my ide",                     "intent": "recall"},
    {"text": "what did i save as my editor",       "intent": "recall"},
    {"text": "recall my project folder",           "intent": "recall"},
    {"text": "what do you remember about me",      "intent": "recall"},
    {"text": "show me my saved preferences",       "intent": "recall"},
    {"text": "what is my preferred browser",       "intent": "recall"},
    {"text": "tell me what you know about me",     "intent": "recall"},

    # ── volume_up ─────────────────────────────────────────────────────
    {"text": "turn the volume up",                 "intent": "volume_up"},
    {"text": "increase volume",                    "intent": "volume_up"},
    {"text": "louder",                             "intent": "volume_up"},
    {"text": "volume up",                          "intent": "volume_up"},
    {"text": "raise the volume",                   "intent": "volume_up"},
    {"text": "make it louder",                     "intent": "volume_up"},
    {"text": "turn it up",                         "intent": "volume_up"},
    {"text": "boost the volume",                   "intent": "volume_up"},

    # ── volume_down ───────────────────────────────────────────────────
    {"text": "turn the volume down",               "intent": "volume_down"},
    {"text": "decrease volume",                    "intent": "volume_down"},
    {"text": "quieter",                            "intent": "volume_down"},
    {"text": "volume down",                        "intent": "volume_down"},
    {"text": "lower the volume",                   "intent": "volume_down"},
    {"text": "make it quieter",                    "intent": "volume_down"},
    {"text": "turn it down",                       "intent": "volume_down"},
    {"text": "reduce volume",                      "intent": "volume_down"},
    {"text": "softer please",                      "intent": "volume_down"},

    # ── mute ──────────────────────────────────────────────────────────
    {"text": "mute",                               "intent": "mute"},
    {"text": "mute the volume",                    "intent": "mute"},
    {"text": "silence",                            "intent": "mute"},
    {"text": "turn off sound",                     "intent": "mute"},
    {"text": "unmute",                             "intent": "mute"},
    {"text": "toggle mute",                        "intent": "mute"},

    # ── screenshot ────────────────────────────────────────────────────
    {"text": "take a screenshot",                  "intent": "screenshot"},
    {"text": "capture the screen",                 "intent": "screenshot"},
    {"text": "screenshot",                         "intent": "screenshot"},
    {"text": "snap the screen",                    "intent": "screenshot"},
    {"text": "take a screen capture",              "intent": "screenshot"},
    {"text": "grab a screenshot",                  "intent": "screenshot"},
    {"text": "capture my screen",                  "intent": "screenshot"},
    {"text": "save a screenshot",                  "intent": "screenshot"},

    # ── shutdown ──────────────────────────────────────────────────────
    {"text": "shutdown the computer",              "intent": "shutdown"},
    {"text": "shut down",                          "intent": "shutdown"},
    {"text": "turn off the pc",                    "intent": "shutdown"},
    {"text": "power off",                          "intent": "shutdown"},
    {"text": "shutdown in 10 minutes",             "intent": "shutdown"},
    {"text": "turn off computer in 30 minutes",    "intent": "shutdown"},
    {"text": "shutdown now",                       "intent": "shutdown"},
    {"text": "power down the machine",             "intent": "shutdown"},
    {"text": "shut the computer off",              "intent": "shutdown"},

    # ── restart ───────────────────────────────────────────────────────
    {"text": "restart the computer",               "intent": "restart"},
    {"text": "reboot",                             "intent": "restart"},
    {"text": "restart now",                        "intent": "restart"},
    {"text": "reboot the pc",                      "intent": "restart"},
    {"text": "restart in 5 minutes",               "intent": "restart"},

    # ── sleep ─────────────────────────────────────────────────────────
    {"text": "put the computer to sleep",          "intent": "sleep"},
    {"text": "sleep mode",                         "intent": "sleep"},
    {"text": "hibernate",                          "intent": "sleep"},
    {"text": "suspend the pc",                     "intent": "sleep"},

    # ── system_info ───────────────────────────────────────────────────
    {"text": "show system info",                   "intent": "system_info"},
    {"text": "what is my cpu usage",               "intent": "system_info"},
    {"text": "how much ram am i using",            "intent": "system_info"},
    {"text": "system status",                      "intent": "system_info"},
    {"text": "show performance stats",             "intent": "system_info"},
    {"text": "how is my pc doing",                 "intent": "system_info"},
    {"text": "disk usage",                         "intent": "system_info"},
    {"text": "show me memory usage",               "intent": "system_info"},
    {"text": "what processes are running",         "intent": "system_info"},
    {"text": "resource usage",                     "intent": "system_info"},

    # ── schedule_task ─────────────────────────────────────────────────
    {"text": "remind me in 10 minutes",            "intent": "schedule_task"},
    {"text": "set a reminder for 5 minutes",       "intent": "schedule_task"},
    {"text": "alert me in 30 minutes",             "intent": "schedule_task"},
    {"text": "remind me to take a break in 1 hour","intent": "schedule_task"},
    {"text": "set a timer for 15 minutes",         "intent": "schedule_task"},

    # ── web_search ────────────────────────────────────────────────────
    {"text": "search for python tutorials",        "intent": "web_search"},
    {"text": "google pytorch documentation",       "intent": "web_search"},
    {"text": "search the web for latest news",     "intent": "web_search"},
    {"text": "look up transformer architecture",   "intent": "web_search"},
    {"text": "search for windows 11 tips",         "intent": "web_search"},

    # ── help ──────────────────────────────────────────────────────────
    {"text": "what can you do",                    "intent": "help"},
    {"text": "help",                               "intent": "help"},
    {"text": "show commands",                      "intent": "help"},
    {"text": "what are your features",             "intent": "help"},
    {"text": "list your abilities",                "intent": "help"},
    {"text": "how do i use you",                   "intent": "help"},

    # ── unknown ───────────────────────────────────────────────────────
    {"text": "blah blah blah",                     "intent": "unknown"},
    {"text": "asdfghjkl",                          "intent": "unknown"},
    {"text": "do the thing",                       "intent": "unknown"},
    {"text": "make it happen",                     "intent": "unknown"},
    {"text": "xyz abc",                            "intent": "unknown"},
    {"text": "idk man",                            "intent": "unknown"},
    {"text": "???",                                "intent": "unknown"},
    {"text": "huh",                                "intent": "unknown"},

    # ── greet ─────────────────────────────────────────────────────────
    {"text": "hello",                              "intent": "greet"},
    {"text": "hi",                                 "intent": "greet"},
    {"text": "hey",                                "intent": "greet"},
    {"text": "hey sentinel",                       "intent": "greet"},
    {"text": "hello sentinel",                     "intent": "greet"},
    {"text": "good morning",                       "intent": "greet"},
    {"text": "good afternoon",                     "intent": "greet"},
    {"text": "good evening",                       "intent": "greet"},
    {"text": "hi there",                           "intent": "greet"},
    {"text": "yo",                                 "intent": "greet"},
    {"text": "what is up",                         "intent": "greet"},
    {"text": "howdy",                              "intent": "greet"},
    {"text": "greetings",                          "intent": "greet"},
    {"text": "morning",                            "intent": "greet"},
    {"text": "sup",                                "intent": "greet"},

    # ── goodbye ───────────────────────────────────────────────────────
    {"text": "bye",                                "intent": "goodbye"},
    {"text": "goodbye",                            "intent": "goodbye"},
    {"text": "see you later",                      "intent": "goodbye"},
    {"text": "later",                              "intent": "goodbye"},
    {"text": "take care",                          "intent": "goodbye"},
    {"text": "i am done",                          "intent": "goodbye"},
    {"text": "that is all for now",                "intent": "goodbye"},
    {"text": "exit",                               "intent": "goodbye"},
    {"text": "close sentinel",                     "intent": "goodbye"},
    {"text": "quit sentinel",                      "intent": "goodbye"},
    {"text": "have a good one",                    "intent": "goodbye"},
    {"text": "peace",                              "intent": "goodbye"},
    {"text": "i am finished",                      "intent": "goodbye"},
    {"text": "catch you later",                    "intent": "goodbye"},

    # ── thanks ────────────────────────────────────────────────────────
    {"text": "thank you",                          "intent": "thanks"},
    {"text": "thanks",                             "intent": "thanks"},
    {"text": "cheers",                             "intent": "thanks"},
    {"text": "appreciate it",                      "intent": "thanks"},
    {"text": "nice one",                           "intent": "thanks"},
    {"text": "perfect thanks",                     "intent": "thanks"},
    {"text": "brilliant",                          "intent": "thanks"},
    {"text": "great job",                          "intent": "thanks"},
    {"text": "well done",                          "intent": "thanks"},
    {"text": "that is perfect",                    "intent": "thanks"},
    {"text": "excellent",                          "intent": "thanks"},
    {"text": "thanks a lot",                       "intent": "thanks"},
    {"text": "ta",                                 "intent": "thanks"},
    {"text": "many thanks",                        "intent": "thanks"},

    # ── chitchat ──────────────────────────────────────────────────────
    {"text": "how are you",                        "intent": "chitchat"},
    {"text": "how are you doing",                  "intent": "chitchat"},
    {"text": "what is your name",                  "intent": "chitchat"},
    {"text": "who are you",                        "intent": "chitchat"},
    {"text": "are you an ai",                      "intent": "chitchat"},
    {"text": "tell me a joke",                     "intent": "chitchat"},
    {"text": "say something funny",                "intent": "chitchat"},
    {"text": "what do you think",                  "intent": "chitchat"},
    {"text": "do you like music",                  "intent": "chitchat"},
    {"text": "are you sentient",                   "intent": "chitchat"},
    {"text": "talk to me",                         "intent": "chitchat"},
    {"text": "what is the meaning of life",        "intent": "chitchat"},
    {"text": "do you dream",                       "intent": "chitchat"},
    {"text": "are you smart",                      "intent": "chitchat"},
    {"text": "what can you feel",                  "intent": "chitchat"},
    {"text": "tell me something interesting",      "intent": "chitchat"},
    {"text": "what is your favourite colour",      "intent": "chitchat"},
    {"text": "do you get bored",                   "intent": "chitchat"},

    # ── clarify ───────────────────────────────────────────────────────
    {"text": "what did you just do",               "intent": "clarify"},
    {"text": "explain that",                       "intent": "clarify"},
    {"text": "what does that mean",                "intent": "clarify"},
    {"text": "can you repeat that",                "intent": "clarify"},
    {"text": "i do not understand",                "intent": "clarify"},
    {"text": "what happened",                      "intent": "clarify"},
    {"text": "say that again",                     "intent": "clarify"},
    {"text": "what did you do",                    "intent": "clarify"},
    {"text": "explain what you did",               "intent": "clarify"},
    {"text": "why did you do that",                "intent": "clarify"},
    {"text": "what just happened",                 "intent": "clarify"},

    # ── status ────────────────────────────────────────────────────────
    {"text": "what are you doing",                 "intent": "status"},
    {"text": "are you busy",                       "intent": "status"},
    {"text": "are you ready",                      "intent": "status"},
    {"text": "what is your status",                "intent": "status"},
    {"text": "are you running anything",           "intent": "status"},
    {"text": "what mode are you in",               "intent": "status"},
    {"text": "are you working",                    "intent": "status"},
]


# ──────────────────────────────────────────────────────────────────────────────
# TrainingDB class
# ──────────────────────────────────────────────────────────────────────────────

class TrainingDB:
    """
    Manages the SQLite training database.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite file.  Created automatically if it does not exist.
    """

    def __init__(self, db_path: str = "data/training.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    # ── Schema ────────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS intents (
                name        TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_examples (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                text       TEXT    NOT NULL,
                intent     TEXT    NOT NULL REFERENCES intents(name),
                source     TEXT    NOT NULL DEFAULT 'default',
                created_at TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_examples_intent
                ON training_examples(intent);
        """)
        self._conn.commit()

    # ── Seeding ───────────────────────────────────────────────────────

    def seed_defaults(self, force: bool = False) -> int:
        """
        Populate the database with the built-in examples.

        Parameters
        ----------
        force : bool
            If True, clears existing default examples before re-inserting.
            If False (default), skips if default examples already exist.

        Returns
        -------
        int
            Number of examples inserted.
        """
        if not force:
            existing = self._conn.execute(
                "SELECT COUNT(*) FROM training_examples WHERE source='default'"
            ).fetchone()[0]
            if existing > 0:
                return 0  # already seeded

        now = datetime.utcnow().isoformat()

        if force:
            self._conn.execute(
                "DELETE FROM training_examples WHERE source='default'"
            )

        # Insert intents
        for name, desc in INTENT_META.items():
            self._conn.execute(
                "INSERT OR IGNORE INTO intents(name, description, created_at) VALUES (?,?,?)",
                (name, desc, now),
            )

        # Insert examples
        rows = [
            (ex["text"], ex["intent"], "default", now)
            for ex in DEFAULT_EXAMPLES
        ]
        self._conn.executemany(
            "INSERT INTO training_examples(text, intent, source, created_at) VALUES (?,?,?,?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    # ── Read ──────────────────────────────────────────────────────────

    def get_all(self) -> List[Dict]:
        """Return all examples as a list of {text, intent} dicts."""
        rows = self._conn.execute(
            "SELECT text, intent FROM training_examples ORDER BY intent, id"
        ).fetchall()
        return [{"text": r["text"], "intent": r["intent"]} for r in rows]

    def get_intent_labels(self) -> List[str]:
        """Return sorted list of all intent names that have at least one example."""
        rows = self._conn.execute(
            "SELECT DISTINCT intent FROM training_examples ORDER BY intent"
        ).fetchall()
        return [r["intent"] for r in rows]

    def get_stats(self) -> Dict[str, int]:
        """Return example count per intent."""
        rows = self._conn.execute(
            "SELECT intent, COUNT(*) as n FROM training_examples GROUP BY intent ORDER BY intent"
        ).fetchall()
        return {r["intent"]: r["n"] for r in rows}

    # ── Write ─────────────────────────────────────────────────────────

    def add(self, text: str, intent: str, source: str = "user") -> int:
        """Add a single example. Returns the new row id."""
        now = datetime.utcnow().isoformat()
        # Ensure intent exists
        self._conn.execute(
            "INSERT OR IGNORE INTO intents(name, description, created_at) VALUES (?,?,?)",
            (intent, "", now),
        )
        cur = self._conn.execute(
            "INSERT INTO training_examples(text, intent, source, created_at) VALUES (?,?,?,?)",
            (text, intent, source, now),
        )
        self._conn.commit()
        return cur.lastrowid or 0

    def add_many(self, examples: List[Dict], source: str = "user") -> int:
        """Add multiple {text, intent} examples. Returns count inserted."""
        now = datetime.utcnow().isoformat()
        intents = {ex["intent"] for ex in examples}
        for intent in intents:
            self._conn.execute(
                "INSERT OR IGNORE INTO intents(name, description, created_at) VALUES (?,?,?)",
                (intent, "", now),
            )
        rows = [(ex["text"], ex["intent"], source, now) for ex in examples]
        self._conn.executemany(
            "INSERT INTO training_examples(text, intent, source, created_at) VALUES (?,?,?,?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def delete(self, row_id: int) -> bool:
        """Delete an example by id. Returns True if a row was deleted."""
        cur = self._conn.execute(
            "DELETE FROM training_examples WHERE id=?", (row_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: auto-seed and return data (used by train.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_training_data(db_path: str = "data/training.db"):
    """
    Open the training DB, seed it on first run, and return
    (raw_data, intent_labels) ready for SentinelTrainer.
    """
    db = TrainingDB(db_path)
    inserted = db.seed_defaults()
    if inserted:
        print(f"[TrainingDB] Seeded {inserted} default examples into {db_path}")
    return db.get_all(), db.get_intent_labels()