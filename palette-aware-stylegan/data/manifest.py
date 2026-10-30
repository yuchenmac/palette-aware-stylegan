"""Source identities and split validation for prepared patch datasets."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping


SPLITS = ("train", "val", "test")


def _relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Manifest {field} must be a nonempty POSIX relative path.")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"Manifest {field} must be a canonical relative path.")
    return value


def validate_manifest(document: object) -> Dict[str, Mapping[str, object]]:
    """Validate one manifest and return patch records indexed by relative path.

    A source identity may occur in only one split. This validates the recorded
    identities; it does not infer whether differently named files depict the same
    original print.
    """

    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Expected a dataset manifest with schema_version 1.")
    sources = document.get("sources")
    patches = document.get("patches")
    if not isinstance(sources, list) or not isinstance(patches, list):
        raise ValueError("Manifest sources and patches must be lists.")
    source_splits: Dict[str, str] = {}
    for record in sources:
        if not isinstance(record, dict):
            raise ValueError("Manifest source records must be objects.")
        source_id, split = record.get("source_id"), record.get("split")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Manifest source_id must be a nonempty string.")
        if split not in SPLITS:
            raise ValueError(f"Invalid split for source {source_id!r}.")
        if source_id in source_splits:
            if source_splits[source_id] != split:
                raise ValueError(f"Source {source_id!r} occurs in multiple splits.")
            raise ValueError(f"Duplicate source record {source_id!r}.")
        source_splits[source_id] = split

    indexed: Dict[str, Mapping[str, object]] = {}
    for record in patches:
        if not isinstance(record, dict):
            raise ValueError("Manifest patch records must be objects.")
        path = _relative_path(record.get("path"), "patch path")
        source_id, split = record.get("source_id"), record.get("split")
        if split not in SPLITS or PurePosixPath(path).parts[0] != split:
            raise ValueError(f"Patch {path!r} does not belong to its declared split.")
        if not isinstance(source_id, str) or source_id not in source_splits:
            raise ValueError(f"Patch {path!r} has an unknown source_id.")
        if source_splits[source_id] != split:
            raise ValueError(f"Source {source_id!r} occurs in multiple splits.")
        if path in indexed:
            raise ValueError(f"Duplicate patch path {path!r}.")
        for field in ("top", "left"):
            coordinate = record.get(field)
            if type(coordinate) is not int or coordinate < 0:
                raise ValueError(f"Patch {path!r} requires a nonnegative integer {field}.")
        if "lab_path" in record:
            _relative_path(record["lab_path"], "Lab path")
        indexed[path] = record
    return indexed


def load_manifest(path: Path) -> Dict[str, Mapping[str, object]]:
    """Read and validate all splits before a dataset consumes patch records."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return validate_manifest(json.load(handle))
