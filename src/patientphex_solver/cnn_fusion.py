"""Conservative fusion of CPU-generated PhenoTagger CNN candidates.

The bundled PhenoTagger model is useful as a high-recall proposal generator,
but its raw output contains many ordinary biomedical phrases and nested
duplicates.  This module deliberately performs only deterministic filtering;
the optional API entity judge can be run afterwards when a candidate needs
additional calibration.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .ontology import normalize_surface

JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class CnnFusionConfig:
    """Rules for turning raw CNN detections into extra entity candidates."""

    min_score: float = 0.9997
    min_text_length: int = 6
    max_per_identifier: int = 10
    avoid_existing_overlap: bool = True
    new_identifiers_only: bool = False
    # Optional corpus calibration for CNN-only additions.  A surface is kept
    # when its held-out training precision is at least this threshold.
    surface_precision: dict[str, float] | None = None
    surface_min_precision: float = 0.0
    surface_min_count: int = 1


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
        if rules.surface_precision is not None:
            surface = normalize_surface(str(entity.get("text", "")))
            observed = rules.surface_precision.get(surface)
            if (
                observed is not None
                and rules.surface_min_count > 0
                and observed < rules.surface_min_precision
            ):
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


def build_surface_precision(
    training_rows: list[JsonObject],
    gold_rows: list[JsonObject],
    *,
    min_count: int = 1,
    exclude_pmc_id: str | None = None,
) -> dict[str, float]:
    """Estimate CNN surface precision without using the target article labels.

    Pass ``exclude_pmc_id`` during leave-one-out evaluation to omit the target
    article. The returned map contains only surfaces observed at least
    ``min_count`` times among the CNN candidates.
    """
    gold_by_id = {str(row["pmc_id"]): row for row in gold_rows}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate_row in training_rows:
        if exclude_pmc_id is not None and str(candidate_row["pmc_id"]) == str(
            exclude_pmc_id
        ):
            continue
        gold = gold_by_id.get(str(candidate_row["pmc_id"]))
        if gold is None:
            continue
        gold_keys = {
            (int(entity["offset"]), int(entity["length"]), str(entity["identifier"]))
            for entity in gold.get("entities", [])
            if entity.get("note") != "NO"
        }
        for entity in candidate_row.get("entities", []):
            surface = normalize_surface(str(entity.get("text", "")))
            if not surface:
                continue
            key = (
                int(entity["offset"]),
                int(entity["length"]),
                str(entity["identifier"]),
            )
            counts[surface]["tp"] += key in gold_keys
            counts[surface]["total"] += 1
    return {
        surface: values["tp"] / values["total"]
        for surface, values in counts.items()
        if values["total"] >= min_count
    }


def fuse_cnn_entities(
    base_rows: list[JsonObject],
    cnn_rows: list[JsonObject],
    config: CnnFusionConfig | None = None,
    *,
    surface_precision_by_document: dict[str, dict[str, float]] | None = None,
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
        document_config = config or CnnFusionConfig()
        if surface_precision_by_document is not None:
            document_config = CnnFusionConfig(
                min_score=document_config.min_score,
                min_text_length=document_config.min_text_length,
                max_per_identifier=document_config.max_per_identifier,
                avoid_existing_overlap=document_config.avoid_existing_overlap,
                new_identifiers_only=document_config.new_identifiers_only,
                surface_precision=surface_precision_by_document.get(pmc_id, {}),
                surface_min_precision=document_config.surface_min_precision,
                surface_min_count=document_config.surface_min_count,
            )
        additions = cnn_additions(
            base_entities,
            list(cnn.get("entities", [])),
            document_config,
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
