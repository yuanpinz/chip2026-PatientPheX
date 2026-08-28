from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz, process

HPO_ROOT = "HP:0000118"
_SYNONYM_RE = re.compile(r'^synonym: "((?:[^"\\]|\\.)*)"')
_DASH_TRANSLATION = str.maketrans(
    {"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"}
)


def normalize_surface(text: str, *, collapse_space: bool = True) -> str:
    value = unicodedata.normalize("NFKC", text).translate(_DASH_TRANSLATION).casefold()
    return " ".join(value.split()) if collapse_space else value


@dataclass(slots=True)
class HpoTerm:
    identifier: str
    name: str = ""
    synonyms: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    alt_ids: list[str] = field(default_factory=list)
    obsolete: bool = False
    replaced_by: str | None = None


@dataclass(slots=True)
class LinkResult:
    identifier: str
    matched_alias: str
    score: float
    exact: bool


class HpoOntology:
    def __init__(self, terms: dict[str, HpoTerm], root: str = HPO_ROOT) -> None:
        self.terms = terms
        self.root = root
        children: dict[str, set[str]] = defaultdict(set)
        self.alt_to_primary: dict[str, str] = {}
        for identifier, term in terms.items():
            for parent in term.parents:
                children[parent].add(identifier)
            for alt_id in term.alt_ids:
                self.alt_to_primary[alt_id] = identifier

        descendants: set[str] = set()
        pending = [root]
        while pending:
            identifier = pending.pop()
            if identifier in descendants:
                continue
            descendants.add(identifier)
            pending.extend(children.get(identifier, ()))
        self.descendants = descendants

        aliases: dict[str, set[str]] = defaultdict(set)
        preferred_aliases: dict[str, set[str]] = defaultdict(set)
        for identifier in descendants:
            term = terms.get(identifier)
            if term is None or term.obsolete:
                continue
            if term.name:
                normalized = normalize_surface(term.name)
                aliases[normalized].add(identifier)
                preferred_aliases[normalized].add(identifier)
            for synonym in term.synonyms:
                aliases[normalize_surface(synonym)].add(identifier)
        self.aliases = dict(aliases)
        self.preferred_aliases = dict(preferred_aliases)
        self._alias_choices = list(self.aliases)

    @classmethod
    def from_obo(cls, path: str | Path, root: str = HPO_ROOT) -> HpoOntology:
        terms: dict[str, HpoTerm] = {}
        current: HpoTerm | None = None
        with Path(path).open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if line == "[Term]":
                    if current is not None:
                        terms[current.identifier] = current
                    current = HpoTerm(identifier="")
                    continue
                if line.startswith("["):
                    if current is not None:
                        terms[current.identifier] = current
                    current = None
                    continue
                if current is None:
                    continue
                if not line:
                    if current.identifier:
                        terms[current.identifier] = current
                    current = None
                elif line.startswith("id: "):
                    current.identifier = line[4:]
                elif line.startswith("name: "):
                    current.name = line[6:]
                elif line.startswith("is_a: "):
                    current.parents.append(line[6:].split()[0])
                elif line.startswith("alt_id: "):
                    current.alt_ids.append(line[8:])
                elif line.startswith("synonym: "):
                    match = _SYNONYM_RE.match(line)
                    if match:
                        current.synonyms.append(
                            bytes(match.group(1), "utf-8").decode("unicode_escape")
                            if "\\" in match.group(1)
                            else match.group(1)
                        )
                elif line == "is_obsolete: true":
                    current.obsolete = True
                elif line.startswith("replaced_by: "):
                    current.replaced_by = line[13:].split()[0]
        if current is not None and current.identifier:
            terms[current.identifier] = current
        return cls(terms, root=root)

    def canonical_id(self, identifier: str) -> str:
        value = self.alt_to_primary.get(identifier, identifier)
        term = self.terms.get(value)
        if term is not None and term.obsolete and term.replaced_by:
            return term.replaced_by
        return value

    def resolve_alias(
        self,
        alias: str,
        *,
        id_frequency: dict[str, int] | None = None,
    ) -> str | None:
        normalized = normalize_surface(alias)
        identifiers = self.aliases.get(normalized)
        if not identifiers:
            return None
        preferred = self.preferred_aliases.get(normalized, set()).intersection(identifiers)
        pool = preferred or identifiers
        frequency = id_frequency or {}
        return max(pool, key=lambda identifier: (frequency.get(identifier, 0), identifier))

    def link(
        self,
        mentions: Iterable[str],
        *,
        id_frequency: dict[str, int] | None = None,
        score_cutoff: float = 78.0,
    ) -> LinkResult | None:
        normalized_mentions = [normalize_surface(item) for item in mentions if item.strip()]
        for mention in normalized_mentions:
            identifier = self.resolve_alias(mention, id_frequency=id_frequency)
            if identifier:
                return LinkResult(identifier, mention, 100.0, True)

        best: tuple[str, float] | None = None
        for mention in normalized_mentions:
            matches = process.extract(
                mention,
                self._alias_choices,
                scorer=fuzz.WRatio,
                score_cutoff=score_cutoff,
                limit=5,
            )
            for alias, score, _ in matches:
                adjusted = float(score)
                if len(alias) < 5:
                    adjusted -= 8
                if best is None or adjusted > best[1]:
                    best = (alias, adjusted)
        if best is None or best[1] < score_cutoff:
            return None
        identifier = self.resolve_alias(best[0], id_frequency=id_frequency)
        if identifier is None:
            return None
        return LinkResult(identifier, best[0], best[1], False)
