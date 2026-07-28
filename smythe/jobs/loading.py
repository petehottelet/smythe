"""Manifest loading helpers for the Jobs CLI and Python callers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

from smythe.jobs.models import (
    MAX_MANIFEST_BYTES,
    JobManifestV1,
    ManifestValidationError,
    strict_json_loads,
)


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that never resolves duplicate mapping keys."""


def _construct_strict_mapping(
    loader: _StrictSafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_strict_mapping,
)


def load_manifest(path: str | Path) -> tuple[JobManifestV1, Path]:
    """Load a strict v1 JSON or YAML manifest and return its attachment root."""
    source = Path(path).resolve()
    try:
        initial = source.stat()
        if initial.st_size > MAX_MANIFEST_BYTES:
            raise ManifestValidationError(
                f"manifest exceeds {MAX_MANIFEST_BYTES} UTF-8 bytes: {source}"
            )
        with source.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if opened.st_size != initial.st_size:
                raise ManifestValidationError(f"manifest changed before it could be read: {source}")
            document = stream.read(MAX_MANIFEST_BYTES + 1)
            closed = os.fstat(stream.fileno())
            if closed.st_size != opened.st_size:
                raise ManifestValidationError(f"manifest changed while it was read: {source}")
    except OSError as exc:
        raise ManifestValidationError(f"cannot read manifest {source}: {exc}") from exc
    if len(document) > MAX_MANIFEST_BYTES:
        raise ManifestValidationError(
            f"manifest exceeds {MAX_MANIFEST_BYTES} UTF-8 bytes: {source}"
        )
    try:
        text = document.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestValidationError(f"manifest is not valid UTF-8: {source}") from exc

    try:
        if source.suffix.lower() == ".json":
            payload: Any = strict_json_loads(text)
        else:
            if any(isinstance(event, AliasEvent) for event in yaml.parse(text)):
                raise ManifestValidationError(
                    "manifest YAML aliases are forbidden to prevent expansion"
                )
            payload = yaml.load(text, Loader=_StrictSafeLoader)
    except ManifestValidationError:
        raise
    except (json.JSONDecodeError, yaml.YAMLError, RecursionError, ValueError) as exc:
        raise ManifestValidationError(f"invalid manifest document: {exc}") from exc
    if payload is None:
        raise ManifestValidationError("manifest document is empty")
    return JobManifestV1.from_dict(payload), source.parent
