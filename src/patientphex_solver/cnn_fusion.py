"""Conservative fusion of CPU-generated PhenoTagger CNN candidates.

The bundled PhenoTagger model is useful as a high-recall proposal generator,
but its raw output contains many ordinary biomedical phrases and nested
duplicates.  This module deliberately performs only deterministic filtering;
the optional API entity judge can be run afterwards when a candidate needs
additional calibration.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class CnnFusionConfig:
    """Rules for turning raw CNN detections into extra entity candidates."""

    min_score: float = 0.9997
    min_text_length: int = 6
    max_per_identifier: int = 10
    avoid_existing_overlap: bool = True
    new_identifiers_only: bool = False


def _span(entity: JsonObject) -> tuple[int, int]:
    return int(entity["offset"]), int(entity["length"])


def _key(entity: JsonObject) -> tuple[int, int, str, str | None]:
    return (
        int(entity["offset"]),
        int(entity["length"]),
        str(entity["identifier"]),
        entity.get("note"),
    )


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    left_start, left_length = left
    right_start, right_length = right
    return max(left_start, right_start) < min(
        left_start + left_length,
        right_start + right_length,
    )


def _submission_entity(entity: JsonObject) -> JsonObject:
    """Drop model-only metadata such as the CNN confidence score."""

    return {
        "identifier": str(entity["identifier"]),
        "type": entity.get("type", "Phenotype"),
        "offset": int(entity["offset"]),
        "length": int(entity["length"]),
        "text": str(entity["text"]),
        "note": entity.get("note"),
    }


def _rank(entity: JsonObject) -> tuple[float, int, int, str]:
    return (
        -float(entity.get("score", 0.0)),
        -len(str(entity.get("text", ""))),
        int(entity["offset"]),
        str(entity["identifier"]),
    )


def cnn_additions(
    base_entities: list[JsonObject],
    cnn_entities: list[JsonObject],
    config: CnnFusionConfig | None = None,
) -> list[JsonObject]:
    """Return deterministic, conservatively filtered CNN additions.

    Existing entities are never replaced.  A CNN candidate is rejected when
    its span overlaps an existing candidate, because the downstream benchmark
    scores exact mention boundaries and the CNN output often contains nested
    variants of the same phrase.  Candidates are capped per HPO identifier so
    a spurious high-confidence phrase cannot dominate a document.
    """

    rules = config or CnnFusionConfig()
    existing_keys = {_key(entity) for entity in base_entities}
    existing_spans = [_span(entity) for entity in base_entities]
    existing_identifiers = {
        unit
        for entity in base_entities
        if entity.get("note") != "NO"
        for unit in str(entity["identifier"]).split(";")
    }
    candidates: list[JsonObject] = []
    for entity in cnn_entities:
        try:
            score = float(entity.get("score", 0.0))
            span = _span(entity)
            key = _key(entity)
        except (KeyError, TypeError, ValueError):
            continue
        if score < rules.min_score:
            continue
        if len(str(entity.get("text", ""))) < rules.min_text_length:
            continue
        if key in existing_keys:
            continue
        if rules.avoid_existing_overlap and any(
            _overlaps(span, existing_span) for existing_span in existing_spans
        ):
            continue
        identifier = str(entity["identifier"])
        if rules.new_identifiers_only and identifier in existing_identifiers:
            continue
        candidates.append(entity)

    grouped: dict[str, list[JsonObject]] = defaultdict(list)
    for entity in candidates:
        grouped[str(entity["identifier"])].append(entity)

    additions: list[JsonObject] = []
    for identifier in sorted(grouped):
        ranked = sorted(grouped[identifier], key=_rank)
        additions.extend(ranked[: max(0, rules.max_per_identifier)])
    return sorted(additions, key=lambda entity: (_span(entity), str(entity["identifier"])))


def fuse_cnn_entities(
    base_rows: list[JsonObject],
    cnn_rows: list[JsonObject],
    config: CnnFusionConfig | None = None,
) -> list[JsonObject]:
    """Fuse raw CNN entities into rows while preserving associations."""

    cnn_by_id = {str(row["pmc_id"]): row for row in cnn_rows}
    fused: list[JsonObject] = []
    for base in base_rows:
        pmc_id = str(base["pmc_id"])
        cnn = cnn_by_id.get(pmc_id)
        if cnn is None:
            raise ValueError(f"missing CNN row for PMC {pmc_id}")
        base_entities = list(base.get("entities", []))
        additions = cnn_additions(
            base_entities,
            list(cnn.get("entities", [])),
            config,
        )
        fused.append(
            {
                "pmc_id": base["pmc_id"],
                "pmid": base.get("pmid"),
                "entities": base_entities + [_submission_entity(item) for item in additions],
                "association": list(base.get("association", [])),
            }
        )
    return fused
