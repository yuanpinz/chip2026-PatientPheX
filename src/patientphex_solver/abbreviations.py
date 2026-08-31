from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .entities import merge_entities
from .io import passage_for_offset

JsonObject = dict[str, Any]

_ACRONYM = r"[A-Z][A-Z0-9]*(?:[-/][A-Z0-9]+)*"
_ACRONYM_RE = re.compile(rf"(?<![A-Za-z0-9])({_ACRONYM})(?![A-Za-z0-9])")
_FORWARD_DEFINITION_RE = re.compile(rf"^\s*\(\s*({_ACRONYM})\s*\)")
_REVERSE_DEFINITION_RE = re.compile(rf"({_ACRONYM})\s*\(\s*$")
_CLOSING_PAREN_RE = re.compile(r"^\s*\)")
_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")
_INITIAL_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "type",
    "with",
}


def _entity_key(entity: JsonObject) -> tuple[int, int, str]:
    return (
        int(entity["offset"]),
        int(entity["length"]),
        str(entity["identifier"]),
    )


def _initials(text: str) -> str:
    values: list[str] = []
    for token in _TOKEN_RE.findall(text):
        if token.lower() in _INITIAL_STOP_WORDS:
            continue
        values.append(token if token.isdigit() else token[0])
    return "".join(values).upper()


def _matches_long_form(acronym: str, text: str) -> bool:
    compact = re.sub(r"[^A-Z0-9]", "", acronym.upper())
    return len(compact) >= 2 and compact == _initials(text)


def _definition_acronyms(
    document: JsonObject,
    base_entities: list[JsonObject],
) -> dict[str, str]:
    identifiers: dict[str, set[str]] = defaultdict(set)
    for entity in base_entities:
        if entity.get("note") == "NO" or str(entity.get("identifier")) == "-1":
            continue
        located = passage_for_offset(document, int(entity["offset"]))
        if located is None:
            continue
        _, passage = located
        passage_text = str(passage.get("text", ""))
        local_start = int(entity["offset"]) - int(passage["offset"])
        local_end = local_start + int(entity["length"])
        if not 0 <= local_start <= local_end <= len(passage_text):
            continue

        forward = _FORWARD_DEFINITION_RE.match(passage_text[local_end : local_end + 24])
        if forward and _matches_long_form(forward.group(1), str(entity["text"])):
            identifiers[forward.group(1)].add(str(entity["identifier"]))

        prefix = passage_text[max(0, local_start - 24) : local_start]
        reverse = _REVERSE_DEFINITION_RE.search(prefix)
        if (
            reverse
            and _CLOSING_PAREN_RE.match(passage_text[local_end : local_end + 4])
            and _matches_long_form(reverse.group(1), str(entity["text"]))
        ):
            identifiers[reverse.group(1)].add(str(entity["identifier"]))

    return {
        acronym: next(iter(values))
        for acronym, values in identifiers.items()
        if len(values) == 1
    }


def discover_abbreviation_entities(
    document: JsonObject,
    base_entities: list[JsonObject],
) -> list[JsonObject]:
    """Propose repeated occurrences of explicitly defined phenotype acronyms.

    The result is intentionally a candidate set rather than an automatically
    accepted annotation. Biomedical articles use the same acronym in background
    and patient-specific contexts, so callers should pass these candidates to an
    API entity judge before merging them into a submission.
    """

    acronym_identifiers = _definition_acronyms(document, base_entities)
    existing = {_entity_key(entity) for entity in base_entities}
    additions: list[JsonObject] = []
    for passage in document.get("full_text", []):
        passage_text = str(passage.get("text", ""))
        passage_offset = int(passage["offset"])
        for match in _ACRONYM_RE.finditer(passage_text):
            acronym = match.group(1)
            identifier = acronym_identifiers.get(acronym)
            if identifier is None:
                continue
            entity = {
                "identifier": identifier,
                "type": "Phenotype",
                "offset": passage_offset + match.start(1),
                "length": len(acronym),
                "text": acronym,
                "note": None,
            }
            if _entity_key(entity) not in existing:
                additions.append(entity)
    return merge_entities(additions)
