"""Tests for get_file_tree language labeling behavior."""

from jcodemunch_mcp.parser import Symbol
from jcodemunch_mcp.storage import IndexStore
from jcodemunch_mcp.tools.get_file_tree import get_file_tree


def _flatten_file_nodes(tree_nodes: list[dict]) -> dict[str, dict]:
    """Return {file_path: node} for all file nodes in a tree response."""
    out: dict[str, dict] = {}
    for node in tree_nodes:
        if node.get("type") == "file":
            out[node["path"]] = node
        elif node.get("type") == "dir":
            out.update(_flatten_file_nodes(node.get("children", [])))
    return out


def test_get_file_tree_prefers_symbol_language_over_extension(tmp_path):
    """A .h file with C symbols should show language='c' instead of extension default."""
    store = IndexStore(base_path=str(tmp_path))
    sym = Symbol(
        id="include-api-h::only_c#function",
        file="include/api.h",
        name="only_c",
        qualified_name="only_c",
        kind="function",
        language="c",
        signature="int only_c(void)",
        byte_offset=0,
        byte_length=20,
    )

    store.save_index(
        owner="tree",
        name="demo",
        source_files=["include/api.h", "src/orphan.cpp"],
        symbols=[sym],
        raw_files={
            "include/api.h": "int only_c(void) { return 0; }\n",
            "src/orphan.cpp": "// no symbols here\n",
        },
        languages={"c": 1, "cpp": 1},
    )

    result = get_file_tree("tree/demo", storage_path=str(tmp_path))
    assert "error" not in result
    files = _flatten_file_nodes(result["tree"])
    assert files["include/api.h"]["language"] == "c"
    assert files["include/api.h"]["symbol_count"] == 1


def test_get_file_tree_falls_back_to_extension_without_symbol_language(tmp_path):
    """When a file has no symbols, get_file_tree should infer language from extension."""
    store = IndexStore(base_path=str(tmp_path))
    sym = Symbol(
        id="src-main-cpp::main#function",
        file="src/main.cpp",
        name="main",
        qualified_name="main",
        kind="function",
        language="cpp",
        signature="int main()",
        byte_offset=0,
        byte_length=10,
    )

    store.save_index(
        owner="tree",
        name="demo2",
        source_files=["src/main.cpp", "include/no_symbols.h"],
        symbols=[sym],
        raw_files={
            "src/main.cpp": "int main() { return 0; }\n",
            "include/no_symbols.h": "/* header with no symbols */\n",
        },
        languages={"cpp": 2},
    )

    result = get_file_tree("tree/demo2", storage_path=str(tmp_path))
    assert "error" not in result
    files = _flatten_file_nodes(result["tree"])
    assert files["include/no_symbols.h"]["language"] == "cpp"
    assert files["include/no_symbols.h"]["symbol_count"] == 0


def _make_index_with_n_files(tmp_path, owner, name, n):
    """Helper: save an index with n synthetic source files."""
    store = IndexStore(base_path=str(tmp_path))
    source_files = [f"src/file_{i}.py" for i in range(n)]
    raw_files = {f: f"# file {i}\n" for i, f in enumerate(source_files)}
    store.save_index(
        owner=owner,
        name=name,
        source_files=source_files,
        symbols=[],
        raw_files=raw_files,
        languages={"python": n},
    )
    return source_files


class TestGetFileTreeTruncation:
    """get_file_tree must cap output at max_files and surface a helpful hint."""

    def test_no_truncation_when_under_limit(self, tmp_path):
        _make_index_with_n_files(tmp_path, "trunc", "small", 10)
        result = get_file_tree("trunc/small", storage_path=str(tmp_path))
        assert "error" not in result
        assert result.get("truncated") is None
        assert result["_meta"]["file_count"] == 10

    def test_truncation_fires_at_max_files(self, tmp_path):
        _make_index_with_n_files(tmp_path, "trunc", "big", 600)
        result = get_file_tree("trunc/big", storage_path=str(tmp_path))
        assert "error" not in result
        assert result["truncated"] is True
        assert result["total_file_count"] == 600
        assert result["_meta"]["file_count"] == 500  # default cap

    def test_truncation_hint_mentions_path_prefix(self, tmp_path):
        _make_index_with_n_files(tmp_path, "trunc", "hint", 600)
        result = get_file_tree("trunc/hint", storage_path=str(tmp_path))
        assert "path_prefix" in result["hint"]

    def test_custom_max_files_respected(self, tmp_path):
        _make_index_with_n_files(tmp_path, "trunc", "custom", 100)
        result = get_file_tree("trunc/custom", max_files=50, storage_path=str(tmp_path))
        assert result["truncated"] is True
        assert result["_meta"]["file_count"] == 50
        assert result["total_file_count"] == 100

    def test_max_files_hint_includes_total(self, tmp_path):
        _make_index_with_n_files(tmp_path, "trunc", "total", 600)
        result = get_file_tree("trunc/total", storage_path=str(tmp_path))
        # hint should tell the user they can raise max_files to get everything
        assert "600" in result["hint"]

    def test_path_prefix_scoping_reduces_below_cap(self, tmp_path):
        """With a path_prefix that matches few files, no truncation should occur."""
        store = IndexStore(base_path=str(tmp_path))
        # 550 files in src/, 10 files in lib/
        source_files = [f"src/file_{i}.py" for i in range(550)] + [f"lib/util_{i}.py" for i in range(10)]
        raw_files = {f: "# x\n" for f in source_files}
        store.save_index(
            owner="trunc", name="scoped",
            source_files=source_files, symbols=[],
            raw_files=raw_files, languages={"python": len(source_files)},
        )
        result = get_file_tree("trunc/scoped", path_prefix="lib/", storage_path=str(tmp_path))
        assert "error" not in result
        assert result.get("truncated") is None
        assert result["_meta"]["file_count"] == 10
