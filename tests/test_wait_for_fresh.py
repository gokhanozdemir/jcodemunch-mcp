"""Tests for wait_for_fresh MCP tool registration."""

import pytest
import json

from jcodemunch_mcp.server import list_tools, call_tool
from jcodemunch_mcp.reindex_state import _repo_states, _repo_events, _freshness_mode


@pytest.fixture(autouse=True)
def reset_state():
    """Reset module-level state before each test."""
    _repo_states.clear()
    _repo_events.clear()
    _freshness_mode.clear()
    yield
    _repo_states.clear()
    _repo_events.clear()
    _freshness_mode.clear()


class TestWaitForFreshTool:
    @pytest.mark.asyncio
    async def test_wait_for_fresh_listed_in_tools(self):
        tools = await list_tools()
        tool_names = [t.name for t in tools]
        assert "wait_for_fresh" in tool_names

    @pytest.mark.asyncio
    async def test_wait_for_fresh_has_repo_param(self):
        tools = await list_tools()
        wait_tool = next(t for t in tools if t.name == "wait_for_fresh")
        props = wait_tool.inputSchema.get("properties", {})
        assert "repo" in props

    @pytest.mark.asyncio
    async def test_call_wait_for_fresh_when_fresh(self):
        from jcodemunch_mcp.reindex_state import mark_reindex_done
        mark_reindex_done("local/test", {"symbol_count": 42})
        result = await call_tool("wait_for_fresh", {"repo": "local/test", "timeout_ms": 100})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["status"] == "fresh"


class TestMetaStalenessFields:
    @pytest.mark.asyncio
    async def test_meta_has_index_stale_field(self):
        result = await call_tool("get_repo_outline", {"repo": "local/nonexistent"})
        data = json.loads(result[0].text)
        assert "_meta" in data
        assert "index_stale" in data["_meta"]

    @pytest.mark.asyncio
    async def test_meta_has_reindexing_field(self):
        result = await call_tool("get_repo_outline", {"repo": "local/nonexistent"})
        data = json.loads(result[0].text)
        assert "reindexing" in data["_meta"]

    @pytest.mark.asyncio
    async def test_meta_reindexing_false_when_idle(self):
        result = await call_tool("get_repo_outline", {"repo": "local/nonexistent"})
        data = json.loads(result[0].text)
        assert data["_meta"]["reindexing"] is False


class TestStrictFreshnessMode:
    @pytest.mark.asyncio
    async def test_strict_waits_for_reindex(self):
        from jcodemunch_mcp.reindex_state import mark_reindex_start, set_freshness_mode
        set_freshness_mode("strict")
        try:
            # Start a reindex (but don't complete it)
            mark_reindex_start("local/test")
            # In strict mode, the call should wait briefly and return with reindexing=True in meta
            result = await call_tool("get_repo_outline", {"repo": "local/test"})
            data = json.loads(result[0].text)
            # In strict mode, await_freshness_if_strict should have waited
            # The meta should show reindexing state
            assert "_meta" in data
            assert "reindexing" in data["_meta"]
        finally:
            set_freshness_mode("relaxed")

