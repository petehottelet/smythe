"""Agent Skills — SKILL.md directories as a source of capability tags.

A skill is a directory holding a ``SKILL.md`` file: YAML frontmatter that
names and describes the skill, then instructions in Markdown. Discovery reads
that one file per skill. It never runs a skill's scripts, imports from its
directory or makes a network request. A discovered skill becomes a capability
tag through the registry's mapper and nothing else: its declared
``allowed-tools`` are recorded but never granted, and its instructions are
never put in a prompt.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from smythe.skills import SkillRef

__all__ = ["AgentSkill", "AgentSkillError", "AgentSkillsProvider", "parse_skill_md"]

SKILL_FILE = "SKILL.md"
SOURCE = "agent-skills"
MAX_SKILL_BYTES = 64 * 1024
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 1024
MAX_COMPATIBILITY_CHARS = 500
_MAX_DETAIL_CHARS = 500
_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_FRONTMATTER = re.compile(r"---[ \t]*\n(.*?)^---[ \t]*(?:\n|\Z)", re.DOTALL | re.MULTILINE)
_KNOWN_FIELDS = frozenset({
    "name", "description", "license", "compatibility", "metadata", "allowed-tools", "version",
})


class AgentSkillError(ValueError):
    """A SKILL.md file or skills directory that cannot be used."""


@dataclass(frozen=True)
class AgentSkill:
    """One parsed SKILL.md: its frontmatter and digests, never its instructions.

    ``version`` is ``"sha256:"`` and the hex digest of the whole file after
    CRLF line endings become LF, so any edit but a line-ending change gives a
    new version. ``body_sha256`` is the hex digest of the instructions alone.
    ``declared_version`` and ``declared_allowed_tools`` are recorded as the
    file states them; Smythe never acts on either, and a declared tool is
    never granted. ``skill_id`` is the skill's directory name, never a path,
    and ``ignored_fields`` names frontmatter keys this release does not read.
    """

    name: str
    description: str
    version: str
    body_sha256: str
    skill_id: str
    declared_version: str | None = None
    declared_allowed_tools: tuple[str, ...] = ()
    license: str | None = None
    compatibility: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict, hash=False)
    ignored_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "declared_allowed_tools", tuple(self.declared_allowed_tools))
        object.__setattr__(self, "ignored_fields", tuple(self.ignored_fields))


# Each skill by name with the file it was read from, and each agent's skill names.
_Snapshot = tuple[Mapping[str, tuple[Path, AgentSkill]], Mapping[str, tuple[str, ...]]]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split(data: bytes, where: str) -> tuple[bytes, str, str]:
    """Return the CRLF-normalized bytes, frontmatter text and instruction body."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("SKILL.md content must be bytes")
    if len(data) > MAX_SKILL_BYTES:
        raise AgentSkillError(f"{where}: SKILL.md exceeds the {MAX_SKILL_BYTES}-byte limit")
    normalized = bytes(data).replace(b"\r\n", b"\n")
    try:
        text = normalized.decode("utf-8").removeprefix("\ufeff")
    except UnicodeDecodeError:
        raise AgentSkillError(f"{where}: SKILL.md is not UTF-8 text") from None
    match = _FRONTMATTER.match(text)
    if match is None:
        raise AgentSkillError(f"{where}: SKILL.md must start with YAML frontmatter between '---' lines")
    return normalized, match.group(1), text[match.end():]


def _shown(value: Any) -> str:
    """Describe a refused value briefly; only short text is repeated as it is.

    YAML aliases share one object many times over, so a few hundred bytes can
    build a list whose ``repr`` runs to gigabytes.
    """
    if value is None or isinstance(value, str) and len(value) <= 2 * MAX_NAME_CHARS:
        return repr(value)
    if isinstance(value, str):
        return f"text of {len(value)} characters"
    return type(value).__name__


def _clipped(error: Exception) -> str:
    """A loader's message, cut short: it can quote a tag or alias the whole file long."""
    detail = str(error)
    return detail if len(detail) <= _MAX_DETAIL_CHARS else detail[:_MAX_DETAIL_CHARS] + " ..."


def _recorded(value: Any, where: str, what: str) -> str:
    try:
        return str(value)
    except ValueError:  # an integer with more digits than Python writes out as text
        raise AgentSkillError(f"{where}: {what} is a number too long to record") from None


def _text(fields: dict[Any, Any], key: str, where: str, limit: int | None = None) -> str | None:
    value = fields.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AgentSkillError(f"{where}: {key} must be text")
    if limit is not None and len(value) > limit:
        raise AgentSkillError(f"{where}: {key} is longer than {limit} characters")
    return value


def parse_skill_md(data: bytes, *, skill_id: str | None = None) -> AgentSkill:
    """Parse one SKILL.md file's bytes into an :class:`AgentSkill`.

    Only the frontmatter is read as YAML, with ``yaml.safe_load``, and it must
    be a mapping; the file may hold at most 64 KiB. ``name`` must be 1 to 64
    lowercase letters, digits and single hyphens and equal *skill_id*, the
    skill's directory name, which defaults to the name. ``description`` is
    required, at most 1024 characters. ``license`` and ``compatibility`` are
    text, ``metadata`` maps text to text, ``allowed-tools`` is space-separated
    text or a list, and ``version`` is recorded as text. Any problem, frontmatter
    that YAML cannot load included, raises :class:`AgentSkillError` naming the
    field, with a message kept short.
    """
    where = "SKILL.md" if skill_id is None else f"Skill {skill_id!r}"
    normalized, frontmatter, body = _split(data, where)
    try:
        fields = yaml.safe_load(frontmatter)
    except yaml.YAMLError as error:
        raise AgentSkillError(f"{where}: frontmatter is not valid YAML: {_clipped(error)}") from None
    except RecursionError:
        raise AgentSkillError(f"{where}: frontmatter nests too deeply to load") from None
    except ValueError as error:  # a date that does not exist, or too many digits
        raise AgentSkillError(f"{where}: frontmatter holds a value YAML cannot load: {_clipped(error)}") from None
    if not isinstance(fields, dict):
        raise AgentSkillError(f"{where}: frontmatter must be a YAML mapping, not {type(fields).__name__}")
    name = fields.get("name")
    if not isinstance(name, str) or len(name) > MAX_NAME_CHARS or not _NAME.fullmatch(name):
        raise AgentSkillError(
            f"{where}: name must be 1 to {MAX_NAME_CHARS} lowercase letters, digits and single "
            f"hyphens, not starting or ending with a hyphen; got {_shown(name)}"
        )
    if skill_id is not None and name != skill_id:
        raise AgentSkillError(f"{where}: name {name!r} must equal its directory name")
    description = _text(fields, "description", where, MAX_DESCRIPTION_CHARS)
    if description is None or not description.strip():
        raise AgentSkillError(f"{where}: description is required")
    metadata = fields.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()):
        raise AgentSkillError(f"{where}: metadata must map text keys to text values")
    tools = fields.get("allowed-tools")
    if tools is None:
        tools = []
    elif isinstance(tools, str):
        tools = tools.split()
    elif not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        raise AgentSkillError(f"{where}: allowed-tools must be text or a list of text")
    version = fields.get("version")
    if isinstance(version, bool) or not isinstance(version, (str, int, float, type(None))):
        raise AgentSkillError(f"{where}: version must be text")
    return AgentSkill(
        name=name, description=description, version="sha256:" + _sha256(normalized),
        body_sha256=_sha256(body.encode("utf-8")), skill_id=name if skill_id is None else skill_id,
        declared_version=None if version is None else _recorded(version, where, "version"),
        declared_allowed_tools=tuple(tools), license=_text(fields, "license", where),
        compatibility=_text(fields, "compatibility", where, MAX_COMPATIBILITY_CHARS),
        metadata=metadata,
        ignored_fields=tuple(sorted(_recorded(key, where, "a frontmatter key")
                                    for key in fields if key not in _KNOWN_FIELDS)),
    )


def _read(path: Path, where: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(MAX_SKILL_BYTES + 1)
    except OSError as error:
        raise AgentSkillError(f"{where}: SKILL.md cannot be read ({error.strerror})") from None


def _scan(root: Path) -> list[tuple[Path, AgentSkill]]:
    """Parse ``<root>/*/SKILL.md``, refusing a file that resolves outside *root*."""
    try:
        base = root.resolve(strict=True)
        entries = sorted(os.listdir(base))
    except OSError:
        raise AgentSkillError(f"Skills root {str(root)!r} is not a readable directory") from None
    found = []
    for skill_id in entries:
        directory = base / skill_id
        where = f"Skill {skill_id!r}"
        try:
            # Listing the directory matches the file name exactly, on
            # case-insensitive file systems too.
            if not directory.is_dir() or SKILL_FILE not in os.listdir(directory):
                continue
            path = (directory / SKILL_FILE).resolve(strict=True)
        except OSError as error:
            raise AgentSkillError(f"{where}: SKILL.md cannot be read ({error.strerror})") from None
        if not path.is_relative_to(base):
            raise AgentSkillError(f"{where}: SKILL.md resolves outside its skills root")
        found.append((path, parse_skill_md(_read(path, where), skill_id=skill_id)))
    return found


class AgentSkillsProvider:
    """SkillProvider over Agent Skills directories, with skills roots per agent id.

        provider = AgentSkillsProvider({"writer": ["skills"]})
        registry = Registry(skill_provider=provider)

    Each root holds one directory per skill, ``<root>/<name>/SKILL.md``.
    Discovery runs when the provider is built and keeps a frozen snapshot;
    only :meth:`refresh` discovers again. A ``SKILL.md`` that resolves
    outside its root, and one name defined by two directories, are refused.
    Each skill is listed as ``SkillRef(name, version=<digest>,
    source="agent-skills")`` whose metadata holds the description, directory
    name, declared allowed tools and declared version: no path and no
    instructions.
    """

    def __init__(self, skills: Mapping[str, Sequence[str | os.PathLike[str]]]) -> None:
        if not isinstance(skills, Mapping):
            raise TypeError("skills must map agent ids to sequences of skills roots")
        roots: dict[str, tuple[Path, ...]] = {}
        for agent_id, paths in skills.items():
            if not isinstance(agent_id, str) or not agent_id:
                raise TypeError("skills must be keyed by non-empty agent ids")
            if isinstance(paths, (str, bytes, os.PathLike)) or not isinstance(paths, Sequence):
                raise TypeError(f"skills[{agent_id!r}] must be a sequence of skills roots")
            roots[agent_id] = tuple(Path(path) for path in paths)
        self._roots = MappingProxyType(roots)
        self._snapshot: _Snapshot = self._discover()

    def _discover(self) -> _Snapshot:
        scanned: dict[Path, list[tuple[Path, AgentSkill]]] = {}
        skills: dict[str, tuple[Path, AgentSkill]] = {}
        agents: dict[str, tuple[str, ...]] = {}
        for agent_id, roots in self._roots.items():
            names: set[str] = set()
            for root in roots:
                key = root.resolve()
                if key not in scanned:
                    scanned[key] = _scan(root)
                for path, skill in scanned[key]:
                    if skill.name in skills and skills[skill.name][0] != path:
                        raise AgentSkillError(
                            f"Skill name {skill.name!r} is defined in more than one skills root"
                        )
                    skills[skill.name] = (path, skill)
                    names.add(skill.name)
            agents[agent_id] = tuple(sorted(names))
        return MappingProxyType(skills), MappingProxyType(agents)

    def list_agent_skills(self, agent_id: str) -> list[SkillRef]:
        skills, agents = self._snapshot
        return [_skill_ref(skills[name][1]) for name in agents.get(agent_id, ())]

    def skill(self, name: str) -> AgentSkill:
        """The discovered record for skill *name*; ``KeyError`` when there is none."""
        skills, _ = self._snapshot
        if name not in skills:
            raise KeyError(f"No discovered skill named {name!r}")
        return skills[name][1]

    def refresh(self) -> list[str]:
        """Discover again; return the sorted names of skills that changed.

        A skill changed when it was added or removed, when its digest
        changed, or when it is now listed for other agents. A failed
        discovery raises and keeps the previous snapshot. A ``Registry``
        caches what it hydrated, so call its ``refresh_all_capabilities()``
        afterwards.
        """
        before = self._holdings(self._snapshot)
        self._snapshot = self._discover()
        after = self._holdings(self._snapshot)
        return sorted(name for name in before.keys() | after.keys()
                      if before.get(name) != after.get(name))

    @staticmethod
    def _holdings(snapshot: _Snapshot) -> dict[str, tuple[str, frozenset[str]]]:
        skills, agents = snapshot
        return {name: (skill.version, frozenset(agent for agent, names in agents.items() if name in names))
                for name, (_, skill) in skills.items()}

    def instructions(self, name: str) -> tuple[str, str]:
        """Return skill *name*'s instruction body and its SHA-256 hex digest.

        The file is read again and must still have the digest discovery
        recorded; otherwise :class:`AgentSkillError` asks for :meth:`refresh`.
        Swarm, its executors and planners never call this: the text reaches a
        prompt only if your code puts it there.
        """
        skills, _ = self._snapshot
        if name not in skills:
            raise KeyError(f"No discovered skill named {name!r}")
        path, skill = skills[name]
        where = f"Skill {skill.skill_id!r}"
        normalized, _, body = _split(_read(path, where), where)
        if "sha256:" + _sha256(normalized) != skill.version:
            raise AgentSkillError(f"{where}: SKILL.md changed since discovery; call refresh()")
        return body, skill.body_sha256


def _skill_ref(skill: AgentSkill) -> SkillRef:
    return SkillRef(name=skill.name, version=skill.version, source=SOURCE, metadata={
        "description": skill.description, "skill_id": skill.skill_id,
        "declared_allowed_tools": list(skill.declared_allowed_tools),
        "declared_version": skill.declared_version,
    })
