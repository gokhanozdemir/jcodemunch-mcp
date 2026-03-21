"""SQLite WAL storage backend for code indexes.

Replaces monolithic JSON files with per-repo SQLite databases.
WAL mode enables concurrent readers + single writer with delta writes.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .index_store import CodeIndex, INDEX_VERSION, _file_hash
from ..parser.symbols import Symbol

logger = logging.getLogger(__name__)

# SQL to create tables and indexes
_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS symbols (
    id         TEXT PRIMARY KEY,
    file       TEXT NOT NULL,
    name       TEXT NOT NULL,
    kind       TEXT,
    signature  TEXT,
    summary    TEXT,
    docstring  TEXT,
    line       INTEGER,
    end_line   INTEGER,
    byte_offset INTEGER,
    byte_length INTEGER,
    parent     TEXT,
    data       TEXT
);

CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);

CREATE TABLE IF NOT EXISTS files (
    path       TEXT PRIMARY KEY,
    hash       TEXT,
    mtime_ns   INTEGER,
    language   TEXT,
    summary    TEXT,
    blob_sha   TEXT,
    imports    TEXT
);
"""

# Pragmas set on every connection open
_PRAGMAS = [
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA wal_autocheckpoint = 1000",
    "PRAGMA cache_size = -8000",
    "PRAGMA busy_timeout = 5000",
]

# Keys stored in the meta table
_META_KEYS = [
    "repo", "owner", "name", "indexed_at", "index_version",
    "git_head", "source_root", "display_name",
    "languages", "context_metadata",
]


class SQLiteIndexStore:
    """Storage backend using SQLite WAL for code indexes.

    One .db file per repo at {base_path}/{slug}.db.
    Content cache remains as individual files at {base_path}/{slug}/.
    """

    def __init__(self, base_path: Optional[str] = None) -> None:
        """Initialize store.

        Args:
            base_path: Base directory for storage. Defaults to ~/.code-index/
        """
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path.home() / ".code-index"
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ── Connection helpers ──────────────────────────────────────────

    def _db_path(self, owner: str, name: str) -> Path:
        """Path to the SQLite database file for a repo."""
        raise NotImplementedError

    def _connect(self, db_path: Path) -> sqlite3.Connection:
        """Open a connection with WAL pragmas and schema ensured."""
        raise NotImplementedError

    def checkpoint_and_close(self, owner: str, name: str) -> None:
        """Compact WAL file on graceful shutdown. Call from server shutdown hook."""
        raise NotImplementedError
        # Implementation: conn.execute("PRAGMA wal_checkpoint(TRUNCATE)"); conn.close()

    def get_file_languages(self, owner: str, name: str) -> dict[str, str]:
        """Query only the files table for path→language mapping.
        Avoids loading the full index when only file_languages is needed."""
        raise NotImplementedError
        # Implementation: SELECT path, language FROM files WHERE language != ''

    def get_symbol_by_id(self, owner: str, name: str, symbol_id: str) -> Optional[dict]:
        """Query a single symbol by ID directly from SQLite.
        Avoids loading the full index for get_symbol_content."""
        raise NotImplementedError
        # Implementation: SELECT * FROM symbols WHERE id = ?

    def has_file(self, owner: str, name: str, file_path: str) -> bool:
        """Check if a file exists in the index without loading the full index."""
        raise NotImplementedError
        # Implementation: SELECT 1 FROM files WHERE path = ?

    # ── Public API (mirrors IndexStore) ─────────────────────────────

    def save_index(
        self,
        owner: str,
        name: str,
        source_files: list[str],
        symbols: list[Symbol],
        raw_files: dict[str, str],
        languages: Optional[dict[str, int]] = None,
        file_hashes: Optional[dict[str, str]] = None,
        git_head: str = "",
        file_summaries: Optional[dict[str, str]] = None,
        source_root: str = "",
        file_languages: Optional[dict[str, str]] = None,
        display_name: str = "",
        imports: Optional[dict[str, list[dict]]] = None,
        context_metadata: Optional[dict] = None,
        file_blob_shas: Optional[dict[str, str]] = None,
        file_mtimes: Optional[dict[str, float]] = None,
    ) -> CodeIndex:
        """Save a full index to SQLite. Replaces all existing data."""
        raise NotImplementedError

    def load_index(self, owner: str, name: str) -> Optional[CodeIndex]:
        """Load index from SQLite, constructing a CodeIndex dataclass."""
        raise NotImplementedError

    def has_index(self, owner: str, name: str) -> bool:
        """Return True if a .db file exists for this repo."""
        raise NotImplementedError

    def incremental_save(
        self,
        owner: str,
        name: str,
        changed_files: list[str],
        new_files: list[str],
        deleted_files: list[str],
        new_symbols: list[Symbol],
        raw_files: dict[str, str],
        languages: Optional[dict[str, int]] = None,
        git_head: str = "",
        file_summaries: Optional[dict[str, str]] = None,
        file_languages: Optional[dict[str, str]] = None,
        imports: Optional[dict[str, list[dict]]] = None,
        context_metadata: Optional[dict] = None,
        file_blob_shas: Optional[dict[str, str]] = None,
        file_hashes: Optional[dict[str, str]] = None,
        file_mtimes: Optional[dict[str, float]] = None,
    ) -> Optional[CodeIndex]:
        """Incrementally update an existing index (delta write)."""
        raise NotImplementedError

    def detect_changes_with_mtimes(
        self,
        owner: str,
        name: str,
        current_mtimes: dict[str, float],
        hash_fn: Callable[[str], str],
    ) -> tuple[list[str], list[str], list[str], dict[str, str], dict[str, float]]:
        """Fast-path change detection using mtimes, falling back to hash."""
        raise NotImplementedError

    def detect_changes(
        self,
        owner: str,
        name: str,
        current_files: dict[str, str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Detect changed, new, and deleted files by comparing hashes."""
        raise NotImplementedError

    def detect_changes_from_hashes(
        self,
        owner: str,
        name: str,
        current_hashes: dict[str, str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Detect changes from precomputed hashes."""
        raise NotImplementedError

    def list_repos(self) -> list[dict]:
        """List all indexed repositories (scans .db and .json files)."""
        raise NotImplementedError

    def delete_index(self, owner: str, name: str) -> bool:
        """Delete a repo's .db, .db-wal, .db-shm, and content dir."""
        raise NotImplementedError

    def get_symbol_content(
        self, owner: str, name: str, symbol_id: str,
        _index: Optional[CodeIndex] = None,
    ) -> Optional[str]:
        """Read symbol source using stored byte offsets from content cache."""
        raise NotImplementedError

    def get_file_content(
        self, owner: str, name: str, file_path: str,
        _index: Optional[CodeIndex] = None,
    ) -> Optional[str]:
        """Read a cached file's full content."""
        raise NotImplementedError

    # ── Content cache helpers (reused from IndexStore) ──────────────

    def _content_dir(self, owner: str, name: str) -> Path:
        """Path to raw content directory."""
        raise NotImplementedError

    def _safe_content_path(self, content_dir: Path, relative_path: str) -> Optional[Path]:
        """Resolve a content path and ensure it stays within content_dir."""
        raise NotImplementedError

    def _write_cached_text(self, path: Path, content: str) -> None:
        """Write cached text without newline translation."""
        raise NotImplementedError

    def _read_cached_text(self, path: Path) -> Optional[str]:
        """Read cached text without newline normalization."""
        raise NotImplementedError

    # ── Internal helpers ────────────────────────────────────────────

    def _symbol_to_row(self, symbol: Symbol) -> tuple:
        """Convert a Symbol to a row tuple for INSERT."""
        raise NotImplementedError

    def _symbol_dict_to_row(self, d: dict) -> tuple:
        """Convert a serialized symbol dict to a row tuple for INSERT."""
        raise NotImplementedError

    def _row_to_symbol_dict(self, row: sqlite3.Row) -> dict:
        """Convert a database row to a symbol dict (matches CodeIndex.symbols format)."""
        raise NotImplementedError

    def _build_index_from_rows(
        self, meta: dict, symbol_rows: list, file_rows: list, owner: str, name: str,
    ) -> CodeIndex:
        """Build a CodeIndex from pre-fetched meta dict, symbol rows, and file rows.
        Used by both load_index and incremental_save to avoid redundant queries."""
        raise NotImplementedError

    def _write_meta(self, conn: sqlite3.Connection, index: CodeIndex) -> None:
        """Write all meta keys for an index."""
        raise NotImplementedError

    def _read_meta(self, conn: sqlite3.Connection) -> dict:
        """Read all meta keys into a dict."""
        raise NotImplementedError

    def _repo_slug(self, owner: str, name: str) -> str:
        """Stable slug for file paths (same as IndexStore._repo_slug)."""
        raise NotImplementedError

    def _safe_repo_component(self, value: str, field_name: str) -> str:
        """Validate/sanitize owner/name for filesystem paths."""
        raise NotImplementedError

    # ── Migration ───────────────────────────────────────────────────

    def migrate_from_json(self, json_path: Path, owner: str, name: str) -> Optional[CodeIndex]:
        """Read a JSON index file and populate the SQLite database."""
        raise NotImplementedError
