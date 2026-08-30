from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .entities import GazetteerExtractor, merge_entities
from .io import passage_for_offset
from .llm import BigModelClient
from .ontology import HpoOntology, normalize_surface

JsonObject = dict[str, Any]

_SYSTEM_PROMPT = """You reproduce the annotation policy of the PatientPheX biomedical benchmark.
For each proposed occurrence, decide whether its exact span and exact HPO identifier should be
annotated as a phenotype mention. The calibration examples are authoritative examples from this
same dataset. Match their scope and boundary policy, not merely general medical plausibility.

Accept explicit abnormal findings, symptoms, malformations, abnormal measurements, developmental
or behavioral findings, and benchmark-supported disease/condition mentions in any article section.
Repeated occurrences and valid nested concepts may each be accepted. Reject a proposal when the
text has a non-phenotype meaning in context, is only normal anatomy/procedure/treatment/gene/variant,
is a broad disease label outside the benchmark policy, has the wrong HPO meaning, or uses a span
boundary unsupported by the calibration examples. A medically reasonable inference is not enough:
the exact words must express the proposed HPO concept.

Return JSON only. Do not add candidates or alter indices."""


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    accepted: bool
    section: str
    text: str
    identifier: str
    hpo_name: str
    context: str
    surface: str


def _identifier_names(ontology: HpoOntology, identifier: str) -> str:
    names: list[str] = []
    for unit in identifier.split(";"):
        term = ontology.terms.get(ontology.canonical_id(unit))
        names.append(term.name if term is not None and term.name else unit)
    return "; ".join(names)


def _context_for_entity(
    document: JsonObject,
    entity: JsonObject,
    *,
    radius: int = 150,
) -> tuple[str, str]:
    located = passage_for_offset(document, int(entity["offset"]))
    if located is None:
        return "UNKNOWN", str(entity.get("text", ""))
    _, passage = located
    passage_text = str(passage.get("text", ""))
    local_start = int(entity["offset"]) - int(passage["offset"])
    local_end = local_start + int(entity["length"])
    start = max(0, local_start - radius)
    end = min(len(passage_text), local_end + radius)
    context = passage_text[start:local_start] + "[[" + passage_text[local_start:local_end]
    context += "]]" + passage_text[local_end:end]
    return str(passage.get("section_type", "UNKNOWN")), " ".join(context.split())


def _gold_units(document: JsonObject) -> tuple[set[tuple[int, int, str]], set[tuple[int, int]]]:
    regular: set[tuple[int, int, str]] = set()
    no_id: set[tuple[int, int]] = set()
    for entity in document.get("entities", []):
        if entity.get("note") == "NO":
            continue
        span = (int(entity["offset"]), int(entity["length"]))
        identifier = str(entity["identifier"])
        if identifier == "-1":
            no_id.add(span)
        else:
            regular.update((*span, unit) for unit in identifier.split(";"))
    return regular, no_id


def _candidate_is_gold(
    entity: JsonObject,
    regular: set[tuple[int, int, str]],
    no_id: set[tuple[int, int]],
) -> bool:
    span = (int(entity["offset"]), int(entity["length"]))
    if span in no_id:
        return True
    units = {
        (*span, unit)
        for unit in str(entity["identifier"]).split(";")
        if unit.startswith("HP:")
    }
    return bool(units) and units.issubset(regular)


def build_calibration_examples(
    training_documents: Iterable[JsonObject],
    extractor: GazetteerExtractor,
    ontology: HpoOntology,
) -> list[CalibrationExample]:
    examples: list[CalibrationExample] = []
    for document in training_documents:
        regular, no_id = _gold_units(document)
        for entity in extractor.extract_document(document):
            if entity.get("note") == "NO":
                continue
            section, context = _context_for_entity(document, entity)
            examples.append(
                CalibrationExample(
                    accepted=_candidate_is_gold(entity, regular, no_id),
                    section=section,
                    text=str(entity["text"]),
                    identifier=str(entity["identifier"]),
                    hpo_name=_identifier_names(ontology, str(entity["identifier"])),
                    context=context,
                    surface=normalize_surface(str(entity["text"])),
                )
            )
    return examples


def _stable_tiebreak(example: CalibrationExample) -> str:
    material = "\x1f".join(
        [
            example.section,
            example.text,
            example.identifier,
            example.context,
            "1" if example.accepted else "0",
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _select_calibration_examples(
    examples: Iterable[CalibrationExample],
    candidates: list[JsonObject],
    document: JsonObject,
    *,
    per_label: int,
) -> list[CalibrationExample]:
    target_surfaces = {normalize_surface(str(item["text"])) for item in candidates}
    target_ids = {
        unit
        for item in candidates
        for unit in str(item["identifier"]).split(";")
    }
    target_sections = {
        _context_for_entity(document, item, radius=0)[0] for item in candidates
    }

    def rank(example: CalibrationExample) -> tuple[int, int, int, str]:
        example_ids = set(example.identifier.split(";"))
        return (
            1 if example.surface in target_surfaces else 0,
            1 if example_ids.intersection(target_ids) else 0,
            1 if example.section in target_sections else 0,
            _stable_tiebreak(example),
        )

    selected: list[CalibrationExample] = []
    source = list(examples)
    for accepted in (True, False):
        matching = [item for item in source if item.accepted is accepted]
        matching.sort(key=rank, reverse=True)
        selected.extend(matching[:per_label])
    return selected


def _candidate_row(
    index: int,
    document: JsonObject,
    entity: JsonObject,
    ontology: HpoOntology,
) -> JsonObject:
    section, context = _context_for_entity(document, entity)
    return {
        "index": index,
        "section": section,
        "text": entity["text"],
        "identifier": entity["identifier"],
        "hpo_name": _identifier_names(ontology, str(entity["identifier"])),
        "context": context,
    }


def _judge_prompt(
    document: JsonObject,
    candidate_rows: list[JsonObject],
    examples: list[CalibrationExample],
) -> str:
    calibration = [
        {
            "label": "ACCEPT" if item.accepted else "REJECT",
            "section": item.section,
            "text": item.text,
            "identifier": item.identifier,
            "hpo_name": item.hpo_name,
            "context": item.context,
        }
        for item in examples
    ]
    return f"""PMC_ID: {document['pmc_id']}

CALIBRATION EXAMPLES FROM OTHER ARTICLES:
{json.dumps(calibration, ensure_ascii=False, separators=(',', ':'))}

CANDIDATE OCCURRENCES:
{json.dumps(candidate_rows, ensure_ascii=False, separators=(',', ':'))}

Return exactly this schema:
{{"accepted_indices":[0,2],"uncertain_indices":[5]}}
Use each supplied index at most once. Put only clear benchmark matches in accepted_indices.
Use uncertain_indices for borderline cases; do not place them in accepted_indices.
"""


def _response_indices(response: Any, key: str, allowed: set[int]) -> set[int]:
    if not isinstance(response, dict):
        return set()
    values = response.get(key, [])
    if not isinstance(values, list):
        return set()
    selected: set[int] = set()
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index in allowed:
            selected.add(index)
    return selected


def judge_entities_with_llm(
    document: JsonObject,
    candidates: list[JsonObject],
    ontology: HpoOntology,
    client: BigModelClient,
    calibration_examples: list[CalibrationExample],
    *,
    batch_size: int = 40,
    calibration_per_label: int = 10,
    include_uncertain: bool = False,
) -> list[JsonObject]:
    positive = [item for item in candidates if item.get("note") != "NO"]
    negated = [item for item in candidates if item.get("note") == "NO"]
    accepted: list[JsonObject] = []
    for start in range(0, len(positive), batch_size):
        batch = positive[start : start + batch_size]
        rows = [
            _candidate_row(index, document, entity, ontology)
            for index, entity in enumerate(batch)
        ]
        examples = _select_calibration_examples(
            calibration_examples,
            batch,
            document,
            per_label=calibration_per_label,
        )
        response = client.chat_json(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _judge_prompt(document, rows, examples)},
            ],
            max_tokens=1800,
        )
        allowed = set(range(len(batch)))
        selected = _response_indices(response, "accepted_indices", allowed)
        if include_uncertain:
            selected.update(_response_indices(response, "uncertain_indices", allowed))
        accepted.extend(batch[index] for index in sorted(selected))
    return merge_entities(accepted, negated)
