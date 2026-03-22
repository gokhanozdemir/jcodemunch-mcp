"""Find all files that reference (import) a given identifier."""

import posixpath
import time
from typing import Optional

from ..storage import IndexStore
from ._utils import resolve_repo


def _find_references_single(
    identifier: str,
    index,
    max_results: int,
    owner: str,
    name: str,
    start: float,
) -> dict:
    """Core logic for a single identifier query. Returns the original flat shape."""
    if index.imports is None:
        return {
            "repo": f"{owner}/{name}",
            "identifier": identifier,
            "references": [],
            "note": "No import data available. Re-index with jcodemunch-mcp >= 1.3.0 to enable find_references.",
            "_meta": {"timing_ms": round((time.perf_counter() - start) * 1000, 1)},
        }

    ident_lower = identifier.lower()
    results = []

    for src_file, file_imports in index.imports.items():
        matches = []
        for imp in file_imports:
            # Match against named imports
            named_match = any(n.lower() == ident_lower for n in imp.get("names", []))
            # Match against specifier stem (e.g. 'IntakeService' in './IntakeService.js')
            spec = imp["specifier"]
            spec_stem = posixpath.splitext(posixpath.basename(spec))[0].lower()
            stem_match = spec_stem == ident_lower

            if named_match or stem_match:
                matches.append({
                    "specifier": spec,
                    "names": imp.get("names", []),
                    "match_type": "named" if named_match else "specifier_stem",
                })

        if matches:
            results.append({"file": src_file, "matches": matches})

    results.sort(key=lambda r: r["file"])

    elapsed = (time.perf_counter() - start) * 1000
    return {
        "repo": f"{owner}/{name}",
        "identifier": identifier,
        "reference_count": len(results),
        "references": results[:max_results],
        "_meta": {
            "timing_ms": round(elapsed, 1),
            "truncated": len(results) > max_results,
            "tip": "For usage-site matching beyond imports, also try search_text or check_references.",
        },
    }


def _find_references_batch(
    identifiers: list[str],
    index,
    max_results: int,
    owner: str,
    name: str,
    start: float,
) -> dict:
    """Batch logic: loop over identifiers, return grouped results array."""
    if index.imports is None:
        return {
            "repo": f"{owner}/{name}",
            "results": [
                {
                    "identifier": ident,
                    "references": [],
                    "note": "No import data available. Re-index with jcodemunch-mcp >= 1.3.0 to enable find_references.",
                }
                for ident in identifiers
            ],
            "_meta": {"timing_ms": round((time.perf_counter() - start) * 1000, 1)},
        }

    results = []

    for identifier in identifiers:
        ident_lower = identifier.lower()
        file_results = []

        for src_file, file_imports in index.imports.items():
            matches = []
            for imp in file_imports:
                named_match = any(n.lower() == ident_lower for n in imp.get("names", []))
                spec = imp["specifier"]
                spec_stem = posixpath.splitext(posixpath.basename(spec))[0].lower()
                stem_match = spec_stem == ident_lower

                if named_match or stem_match:
                    matches.append({
                        "specifier": spec,
                        "names": imp.get("names", []),
                        "match_type": "named" if named_match else "specifier_stem",
                    })

            if matches:
                file_results.append({"file": src_file, "matches": matches})

        file_results.sort(key=lambda r: r["file"])
        results.append({
            "identifier": identifier,
            "reference_count": len(file_results),
            "references": file_results[:max_results],
        })

    elapsed = (time.perf_counter() - start) * 1000
    return {
        "repo": f"{owner}/{name}",
        "results": results,
        "_meta": {"timing_ms": round(elapsed, 1)},
    }


def find_references(
    repo: str,
    identifier: Optional[str] = None,
    max_results: int = 50,
    storage_path: Optional[str] = None,
    identifiers: Optional[list[str]] = None,
) -> dict:
    """Find all indexed files that import or reference an identifier.

    Supports two modes:
    - Singular: pass ``identifier`` to get the original flat response shape.
    - Batch: pass ``identifiers`` (list) to query multiple identifiers at once,
      returning a grouped ``results`` array.

    Args:
        repo: Repository identifier (owner/repo or display name).
        identifier: The symbol/module name to look for (singular mode, e.g. 'bulkImport').
        max_results: Maximum number of results.
        storage_path: Custom storage path.
        identifiers: List of symbol/module names to look for (batch mode).

    Returns:
        Singular mode: dict with flat ``references`` list and _meta envelope.
        Batch mode: dict with ``results`` array (one entry per input identifier).

    Raises:
        ValueError: if neither or both of identifier and identifiers are provided.
    """
    if (identifier is None and identifiers is None) or (identifier is not None and identifiers is not None):
        raise ValueError("Provide exactly one of 'identifier' or 'identifiers', not both and not neither.")

    start = time.perf_counter()
    max_results = max(1, min(max_results, 200))

    try:
        owner, name = resolve_repo(repo, storage_path)
    except ValueError as e:
        return {"error": str(e)}

    store = IndexStore(base_path=storage_path)
    index = store.load_index(owner, name)
    if not index:
        return {"error": f"Repository not indexed: {owner}/{name}"}

    if identifiers is not None:
        return _find_references_batch(identifiers, index, max_results, owner, name, start)
    else:
        return _find_references_single(identifier, index, max_results, owner, name, start)
