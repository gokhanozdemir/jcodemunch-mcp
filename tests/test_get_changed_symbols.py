"""Tests for get_changed_symbols tool."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from jcodemunch_mcp.parser import Symbol
from jcodemunch_mcp.storage import IndexStore
from jcodemunch_mcp.tools.get_changed_symbols import get_changed_symbols
from jcodemunch_mcp.tools.index_folder import index_folder


_DUMMY_SHA = "a" * 40


def _seed_index(tmp_path, storage, source_root="", git_head=_DUMMY_SHA):
    """Seed a minimal index directly (no git, no files needed)."""
    store = IndexStore(base_path=storage)
    sym = Symbol(
        id="src/mod.py::placeholder#function",
        file="src/mod.py",
        name="placeholder",
        qualified_name="placeholder",
        kind="function",
        language="python",
        signature="def placeholder():",
        byte_offset=0, byte_length=20,
    )
    store.save_index(
        owner="owner",
        name="testrepo",
        source_files=["src/mod.py"],
        symbols=[sym],
        raw_files={"src/mod.py": "def placeholder(): pass\n"},
        languages={"python": 1},
        file_languages={"src/mod.py": "python"},
        git_head=git_head,
        source_root=source_root,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: str) -> None:
    subprocess.run(["git"] + args, cwd=cwd, check=True,
                   capture_output=True, stdin=subprocess.DEVNULL)


def _make_git_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str, str]:
    """
    Create a git repo at tmp_path with given files as the initial commit.
    Returns (repo_path, repo_id, storage_path).
    """
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    _git(["init"], str(tmp_path))
    _git(["config", "user.email", "test@test.com"], str(tmp_path))
    _git(["config", "user.name", "Test"], str(tmp_path))
    _git(["add", "-A"], str(tmp_path))
    _git(["commit", "-m", "initial"], str(tmp_path))

    storage = str(tmp_path / ".index")
    result = index_folder(str(tmp_path), use_ai_summaries=False, storage_path=storage)
    repo_id = result.get("repo", str(tmp_path))
    return tmp_path, repo_id, storage


def _commit_changes(repo_path: Path, changes: dict[str, str | None]) -> str:
    """Apply file changes (None = delete) and commit. Return new HEAD SHA."""
    for rel, content in changes.items():
        p = repo_path / rel
        if content is None:
            p.unlink(missing_ok=True)
            _git(["rm", "--ignore-unmatch", rel], str(repo_path))
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            _git(["add", rel], str(repo_path))
    _git(["add", "-A"], str(repo_path))
    _git(["commit", "-m", "changes"], str(repo_path))
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_path), capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_repo(self, tmp_path):
        result = get_changed_symbols("nonexistent/repo", storage_path=str(tmp_path / ".index"))
        assert "error" in result

    def test_github_repo_returns_not_local(self, tmp_path):
        """Repos without source_root (GitHub-indexed) return is_local=False."""
        storage = str(tmp_path / ".index")
        _seed_index(tmp_path, storage, source_root="")  # empty = no local root

        result = get_changed_symbols("owner/testrepo", storage_path=storage)
        assert "error" in result
        assert result.get("is_local") is False

    def test_no_sha_at_index_time(self, tmp_path):
        """If index has no git_head, tool returns error asking for explicit since_sha."""
        storage = str(tmp_path / ".index")
        # Init a real git repo so the tool passes the git-availability check
        _git(["init"], str(tmp_path))
        _git(["config", "user.email", "t@t.com"], str(tmp_path))
        _git(["config", "user.name", "T"], str(tmp_path))
        (tmp_path / "x.py").write_text("x = 1")
        _git(["add", "-A"], str(tmp_path))
        _git(["commit", "-m", "init"], str(tmp_path))

        _seed_index(tmp_path, storage, source_root=str(tmp_path), git_head="")

        result = get_changed_symbols("owner/testrepo", storage_path=storage)
        assert "error" in result

    def test_bad_since_sha(self, tmp_path):
        """An invalid since_sha returns a clear error."""
        repo_path, repo_id, storage = _make_git_repo(tmp_path, {"a.py": "def f(): pass"})
        result = get_changed_symbols(repo_id, since_sha="deadbeef00000000000000000000000000000000", storage_path=storage)
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: symbol change detection
# ---------------------------------------------------------------------------

class TestSymbolDetection:
    def test_added_symbol(self, tmp_path):
        """New function in changed file appears in added_symbols."""
        repo_path, repo_id, storage = _make_git_repo(tmp_path, {
            "src/engine.py": "def run(): pass\n",
        })
        initial_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, check=True,
        ).stdout.strip()

        _commit_changes(repo_path, {
            "src/engine.py": "def run(): pass\n\ndef stop(): pass\n",
        })

        result = get_changed_symbols(repo_id, since_sha=initial_sha, storage_path=storage)
        assert "error" not in result, result
        names = {s["name"] for s in result["added_symbols"]}
        assert "stop" in names

    def test_removed_symbol(self, tmp_path):
        """Deleted function appears in removed_symbols."""
        repo_path, repo_id, storage = _make_git_repo(tmp_path, {
            "src/engine.py": "def run(): pass\n\ndef stop(): pass\n",
        })
        initial_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, check=True,
        ).stdout.strip()

        _commit_changes(repo_path, {
            "src/engine.py": "def run(): pass\n",
        })

        result = get_changed_symbols(repo_id, since_sha=initial_sha, storage_path=storage)
        assert "error" not in result, result
        names = {s["name"] for s in result["removed_symbols"]}
        assert "stop" in names

    def test_modified_symbol(self, tmp_path):
        """Function body change → appears in changed_symbols with change_type=modified."""
        repo_path, repo_id, storage = _make_git_repo(tmp_path, {
            "src/engine.py": 'def run():\n    return 1\n',
        })
        initial_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, check=True,
        ).stdout.strip()

        _commit_changes(repo_path, {
            "src/engine.py": 'def run():\n    return 42\n',
        })

        result = get_changed_symbols(repo_id, since_sha=initial_sha, storage_path=storage)
        assert "error" not in result, result
        modified = [s for s in result["changed_symbols"] if s["change_type"] == "modified"]
        names = {s["name"] for s in modified}
        assert "run" in names

    def test_unchanged_symbol_not_in_results(self, tmp_path):
        """A symbol untouched between commits should not appear in any list."""
        repo_path, repo_id, storage = _make_git_repo(tmp_path, {
            "src/engine.py": "def run(): pass\n\ndef stop(): pass\n",
        })
        initial_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, check=True,
        ).stdout.strip()

        _commit_changes(repo_path, {
            "src/engine.py": "def run(): pass\n\ndef stop(): pass\n\ndef new_fn(): pass\n",
        })

        result = get_changed_symbols(repo_id, since_sha=initial_sha, storage_path=storage)
        assert "error" not in result, result
        all_names = (
            {s["name"] for s in result["added_symbols"]}
            | {s["name"] for s in result["removed_symbols"]}
            | {s["name"] for s in result["changed_symbols"]}
        )
        # run and stop unchanged → should not appear
        assert "run" not in all_names
        assert "stop" not in all_names
        # new_fn is added
        assert "new_fn" in {s["name"] for s in result["added_symbols"]}

    def test_counts_consistent(self, tmp_path):
        """added_count, removed_count, changed_count match list lengths."""
        repo_path, repo_id, storage = _make_git_repo(tmp_path, {
            "src/mod.py": "def alpha(): pass\ndef beta(): pass\n",
        })
        initial_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, check=True,
        ).stdout.strip()

        _commit_changes(repo_path, {
            "src/mod.py": "def alpha(): return 1\ndef gamma(): pass\n",
        })

        result = get_changed_symbols(repo_id, since_sha=initial_sha, storage_path=storage)
        assert result["added_count"] == len(result["added_symbols"])
        assert result["removed_count"] == len(result["removed_symbols"])
        assert result["changed_count"] == len(result["changed_symbols"])

    def test_no_changes(self, tmp_path):
        """Same SHA for both ends returns empty lists."""
        repo_path, repo_id, storage = _make_git_repo(tmp_path, {
            "src/mod.py": "def alpha(): pass\n",
        })
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, check=True,
        ).stdout.strip()

        result = get_changed_symbols(repo_id, since_sha=sha, until_sha=sha, storage_path=storage)
        assert "error" not in result, result
        assert result["added_count"] == 0
        assert result["removed_count"] == 0
        assert result["changed_count"] == 0

    def test_include_blast_radius(self, tmp_path):
        """include_blast_radius=True adds blast_radius field to changed entries."""
        repo_path, repo_id, storage = _make_git_repo(tmp_path, {
            "src/engine.py": "def run(): pass\n",
            "src/main.py": "from src.engine import run\nrun()\n",
        })
        initial_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, check=True,
        ).stdout.strip()

        _commit_changes(repo_path, {
            "src/engine.py": "def run(): return 1\n",
        })

        result = get_changed_symbols(
            repo_id, since_sha=initial_sha,
            include_blast_radius=True, storage_path=storage,
        )
        assert "error" not in result, result
        for entry in result["changed_symbols"] + result["added_symbols"] + result["removed_symbols"]:
            assert "blast_radius" in entry

    def test_suppress_meta(self, tmp_path):
        """suppress_meta=True strips the _meta field."""
        repo_path, repo_id, storage = _make_git_repo(tmp_path, {
            "src/mod.py": "def f(): pass\n",
        })
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, check=True,
        ).stdout.strip()

        result = get_changed_symbols(repo_id, since_sha=sha, until_sha=sha,
                                     suppress_meta=True, storage_path=storage)
        assert "_meta" not in result
