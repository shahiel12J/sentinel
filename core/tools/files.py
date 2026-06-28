"""
Sentinel Tools — File Operations

Handles:
  - search_files      find by extension / time filter / location
  - create_folder     mkdir
  - move_files        bulk move by extension
  - copy_files        bulk copy by extension
  - delete_files      bulk delete
  - rename_file       rename
  - open_file         open with default application
  - read_file         read text content
  - summarize_text    extractive summarization (TF-IDF)
  - list_files        directory listing
"""

import os
import re
import sys
import time
import shutil
import hashlib
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter


# ─────────────────────────────────────────────
# Common folders
# ─────────────────────────────────────────────

def _resolve_location(location: str) -> Path:
    """Map a human-readable location to an absolute path."""
    home = Path.home()
    location = location.lower().strip()

    mapping = {
        "desktop":   home / "Desktop",
        "downloads": home / "Downloads",
        "documents": home / "Documents",
        "pictures":  home / "Pictures",
        "music":     home / "Music",
        "videos":    home / "Videos",
        "temp":      Path(os.environ.get("TEMP", "/tmp")),
        "tmp":       Path(os.environ.get("TEMP", "/tmp")),
        "appdata":   Path(os.environ.get("APPDATA", str(home))),
        "":          Path.cwd(),
        "current":   Path.cwd(),
        "here":      Path.cwd(),
    }
    return mapping.get(location, home / location if location else Path.cwd())


# ─────────────────────────────────────────────
# File search
# ─────────────────────────────────────────────

class FileTools:

    # ── Search ───────────────────────────────────────────────────────

    def search_files(
        self,
        extension:   str = "*",
        time_filter: str = "",
        location:    str = "",
        raw_query:   str = "",
        max_results: int = 200,
    ) -> Tuple[bool, str, List[Path]]:
        """
        Search for files matching criteria.

        Returns (success, message, [paths]).
        """
        base = _resolve_location(location)
        if not base.exists():
            return False, f"Directory not found: {base}", []

        # Build glob pattern
        pattern = f"*.{extension}" if extension and extension != "*" else "*"

        try:
            results: List[Path] = []
            for p in base.rglob(pattern):
                if not p.is_file():
                    continue
                if time_filter == "today" and not _modified_today(p):
                    continue
                elif time_filter == "week" and not _modified_this_week(p):
                    continue
                results.append(p)
                if len(results) >= max_results:
                    break

            # Sort by modification time (newest first)
            results.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            if not results:
                msg = f"No {extension.upper() if extension != '*' else ''} files found"
                if time_filter:
                    msg += f" (filter: {time_filter})"
                return True, msg + ".", []

            # Build human-readable list (show up to 20)
            lines = [f"Found {len(results)} file{'s' if len(results) != 1 else ''}."]
            for p in results[:20]:
                mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                size  = _human_size(p.stat().st_size)
                lines.append(f"  {p.name}  ({size}, {mtime})")
            if len(results) > 20:
                lines.append(f"  … and {len(results) - 20} more.")

            return True, "\n".join(lines), results

        except PermissionError as e:
            return False, f"Permission denied: {e}", []

    # ── Create folder ─────────────────────────────────────────────────

    def create_folder(self, name: str, parent: str = "") -> Tuple[bool, str]:
        """Create a new folder."""
        base = _resolve_location(parent) if parent else Path.cwd()
        target = base / name

        if target.exists():
            return True, f"Folder '{name}' already exists at {target}."
        try:
            target.mkdir(parents=True)
            return True, f"Folder '{name}' created at {target}."
        except Exception as e:
            return False, f"Failed to create folder '{name}': {e}"

    # ── Move files ────────────────────────────────────────────────────

    def move_files(
        self,
        extension:   str,
        destination: str,
        source:      str = "",
    ) -> Tuple[bool, str]:
        """Move all files with given extension to destination folder."""
        src_base = _resolve_location(source)
        dst_base = _resolve_location(destination)

        # If destination doesn't exist, create it in CWD
        if not dst_base.exists():
            dst_base = Path.cwd() / destination
            dst_base.mkdir(parents=True, exist_ok=True)

        pattern = f"*.{extension}" if extension != "*" else "*"
        moved, errors = 0, []

        for p in src_base.glob(pattern):
            if not p.is_file():
                continue
            dest_file = dst_base / p.name
            # Avoid overwriting: append number if conflict
            if dest_file.exists():
                stem, suffix = p.stem, p.suffix
                dest_file = dst_base / f"{stem}_{int(time.time())}{suffix}"
            try:
                shutil.move(str(p), str(dest_file))
                moved += 1
            except Exception as e:
                errors.append(str(e))

        msg = f"Moved {moved} .{extension} file{'s' if moved != 1 else ''} to '{destination}'."
        if errors:
            msg += f"\n{len(errors)} error(s): {errors[0]}"
        return moved > 0 or len(errors) == 0, msg

    # ── Copy files ────────────────────────────────────────────────────

    def copy_files(
        self,
        extension:   str,
        destination: str,
        source:      str = "",
    ) -> Tuple[bool, str]:
        """Copy all files with given extension to destination."""
        src_base = _resolve_location(source)
        dst_base = _resolve_location(destination)

        if not dst_base.exists():
            dst_base = Path.cwd() / destination
            dst_base.mkdir(parents=True, exist_ok=True)

        pattern = f"*.{extension}" if extension != "*" else "*"
        copied, errors = 0, []

        for p in src_base.glob(pattern):
            if not p.is_file():
                continue
            dest_file = dst_base / p.name
            try:
                shutil.copy2(str(p), str(dest_file))
                copied += 1
            except Exception as e:
                errors.append(str(e))

        return copied > 0 or not errors, \
               f"Copied {copied} .{extension} file{'s' if copied != 1 else ''} to '{destination}'."

    # ── Delete files ──────────────────────────────────────────────────

    def delete_files(
        self,
        extension: str,
        location:  str = "",
    ) -> Tuple[bool, str]:
        """Delete files with given extension."""
        base    = _resolve_location(location)
        pattern = f"*.{extension}" if extension != "*" else "*"
        deleted, errors = 0, []

        for p in base.glob(pattern):
            if not p.is_file():
                continue
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                errors.append(str(e))

        return True, f"Deleted {deleted} .{extension} file{'s' if deleted != 1 else ''}."

    # ── Rename ────────────────────────────────────────────────────────

    def rename_file(self, raw_command: str) -> Tuple[bool, str]:
        """Parse and execute a rename command."""
        # "rename X to Y" or "rename X as Y"
        m = re.search(r"rename\s+(.+?)\s+(?:to|as)\s+(.+)", raw_command, re.IGNORECASE)
        if not m:
            return False, "Could not parse rename command. Format: rename <old> to <new>"

        old_name = m.group(1).strip()
        new_name = m.group(2).strip()

        for search_dir in [Path.cwd(), Path.home() / "Desktop", Path.home() / "Downloads"]:
            old_path = search_dir / old_name
            if old_path.exists():
                new_path = search_dir / new_name
                old_path.rename(new_path)
                return True, f"Renamed '{old_name}' to '{new_name}'."

        return False, f"File '{old_name}' not found."

    # ── Open file ─────────────────────────────────────────────────────

    def open_file(self, filename: str, raw_command: str = "") -> Tuple[bool, str]:
        """Open a file with its default application."""
        # Look in common locations
        search_dirs = [
            Path.cwd(),
            Path.home() / "Desktop",
            Path.home() / "Downloads",
            Path.home() / "Documents",
        ]

        for d in search_dirs:
            p = d / filename
            if p.exists():
                try:
                    if sys.platform == "win32":
                        os.startfile(str(p))
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", str(p)])
                    else:
                        subprocess.Popen(["xdg-open", str(p)])
                    return True, f"Opened '{p.name}'."
                except Exception as e:
                    return False, f"Failed to open '{filename}': {e}"

        return False, f"File '{filename}' not found in common locations."

    # ── Read file ─────────────────────────────────────────────────────

    def read_file(self, filename: str, raw_command: str = "") -> Tuple[bool, str]:
        """Read a text file and return its content."""
        search_dirs = [
            Path.cwd(),
            Path.home() / "Desktop",
            Path.home() / "Downloads",
            Path.home() / "Documents",
        ]

        # If raw_command contains a full path, use it
        path_match = re.search(r"[A-Za-z]:[\\\/][^\s]+", raw_command)
        if path_match:
            p = Path(path_match.group(0))
            if p.exists():
                try:
                    return True, p.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    return False, f"Could not read file: {e}"

        for d in search_dirs:
            p = d / filename if filename else None
            if p and p.exists():
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                    # Store for summarizer
                    self._last_file_content = content
                    self._last_file_name    = p.name
                    return True, content[:5000]   # limit for display
                except Exception as e:
                    return False, f"Could not read '{filename}': {e}"

        return False, f"Could not find '{filename}'."

    # ── Summarize ─────────────────────────────────────────────────────

    def summarize_text(self, filename: str = "") -> Tuple[bool, str]:
        """
        Extractive summarization using TF-IDF sentence scoring.
        Operates on the last read file content.
        """
        content = getattr(self, "_last_file_content", None)
        if not content:
            return False, "No file content to summarize. Please read a file first."

        fname = getattr(self, "_last_file_name", filename or "file")
        summary = _extractive_summarize(content, max_sentences=5)

        result = f"Summary of '{fname}':\n\n{summary}"
        return True, result

    # ── List files ────────────────────────────────────────────────────

    def list_files(self, location: str = "") -> Tuple[bool, str]:
        """List files in a directory."""
        base = _resolve_location(location)
        if not base.exists():
            return False, f"Directory not found: {base}"

        try:
            entries = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            lines   = [f"Contents of {base}:", ""]
            dirs, files = [], []

            for e in entries:
                if e.is_dir():
                    dirs.append(e)
                elif e.is_file():
                    files.append(e)

            for d in dirs[:20]:
                lines.append(f"  📁 {d.name}/")
            for f in files[:30]:
                size = _human_size(f.stat().st_size)
                lines.append(f"  📄 {f.name}  ({size})")

            total = len(dirs) + len(files)
            if total > 50:
                lines.append(f"  … {total - 50} more items")
            lines.append(f"\n{len(dirs)} folder(s), {len(files)} file(s)")

            return True, "\n".join(lines)
        except PermissionError:
            return False, f"Permission denied: {base}"


# ─────────────────────────────────────────────
# Extractive summarizer (TF-IDF)
# ─────────────────────────────────────────────

def _tokenize_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _extractive_summarize(text: str, max_sentences: int = 5) -> str:
    sentences = _tokenize_sentences(text)
    if not sentences:
        return text[:500]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # TF: term frequency per sentence
    all_words = _tokenize_words(text)
    word_count = Counter(all_words)
    total_words = max(len(all_words), 1)

    # IDF approximation: penalise very common words
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "it", "this", "that", "are", "was",
        "be", "have", "has", "had", "from", "by", "as", "at", "its", "not"
    }

    def sentence_score(sent: str) -> float:
        words = [w for w in _tokenize_words(sent) if w not in stopwords]
        if not words:
            return 0.0
        return sum(word_count[w] / total_words for w in words) / len(words)

    scored = [(sentence_score(s), i, s) for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top sentences and re-order by original position
    top = sorted(scored[:max_sentences], key=lambda x: x[1])
    return " ".join(s for _, _, s in top)


# ─────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────

def _modified_today(p: Path) -> bool:
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    return mtime.date() == datetime.today().date()


def _modified_this_week(p: Path) -> bool:
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    return mtime >= datetime.now() - timedelta(days=7)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} PB"
