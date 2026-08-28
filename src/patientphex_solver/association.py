from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .io import passage_for_offset
from .llm import BigModelClient

JsonObject = dict[str, Any]

_ASSOCIATION_SYSTEM_PROMPT = """You are an expert biomedical information extraction system.
Associate phenotype entity candidates with the specific patients described in a PMC article.
You must select only from the supplied entity indices. Do not invent entities or HPO IDs.
An association means the phenotype was observed in, diagnosed in, or explicitly attributed to that patient.
Exclude general disease descriptions, family members who are not a listed patient, differential diagnoses,
experimental background, and explicitly negated findings. Return JSON only."""


def _entity_value(entity: JsonObject) -> str:
    identifier = str(entity["identifier"])
    return str(entity["text"]) if identifier == "-1" else identifier


def _passage_index(document: JsonObject, entity: JsonObject) -> int | None:
    located = passage_for_offset(document, int(entity["offset"]))
    return located[0] if located else None


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max(text.rfind(mark, 0, start) for mark in ".!?;\n") + 1
    right_candidates = [text.find(mark, end) for mark in ".!?;\n"]
    right_candidates = [value for value in right_candidates if value >= 0]
    right = min(right_candidates, default=len(text)) + 1
    return left, right


def _candidate_context(document: JsonObject, entity: JsonObject) -> str:
    located = passage_for_offset(document, int(entity["offset"]))
    if located is None:
        return ""
    _, passage = located
    passage_start = int(passage["offset"])
    local_start = int(entity["offset"]) - passage_start
    local_end = local_start + int(entity["length"])
    sentence_start, sentence_end = _sentence_bounds(
        str(passage["text"]), local_start, local_end
    )
    sentence = str(passage["text"])[sentence_start:sentence_end].strip()
    return re.sub(r"\s+", " ", sentence)[:360]


def _patient_context(document: JsonObject, patient: JsonObject) -> list[str]:
    contexts: list[str] = []
    for mention in patient.get("mention", []):
        located = passage_for_offset(document, int(mention["offset"]))
        if located is None:
            continue
        _, passage = located
        passage_text = str(passage["text"])
        local_start = int(mention["offset"]) - int(passage["offset"])
        left = max(0, local_start - 120)
        right = min(len(passage_text), local_start + int(mention["length"]) + 180)
        contexts.append(re.sub(r"\s+", " ", passage_text[left:right]))
    return contexts


def _patient_association_prompt(
    document: JsonObject,
    patient: JsonObject,
    entities: list[JsonObject],
) -> str:
    entity_rows = []
    for fallback_index, entity in enumerate(entities):
        entity_rows.append(
            {
                "index": entity.get("index", fallback_index),
                "passage": _passage_index(document, entity),
                "offset": entity["offset"],
                "length": entity["length"],
                "text": entity["text"],
                "identifier": entity["identifier"],
                "context": _candidate_context(document, entity),
            }
        )

    return f"""PMC_ID: {document['pmc_id']}

TARGET PATIENT:
{json.dumps({
    "patient_id": patient["patient_id"],
    "mentions": patient.get("mention", []),
    "mention_context": _patient_context(document, patient),
}, ensure_ascii=False)}

POSITIVE PHENOTYPE ENTITY CANDIDATES:
{json.dumps(entity_rows, ensure_ascii=False)}

Select all and only candidate indices explicitly attributable to the target patient.
Each candidate includes its containing sentence as context. The same HPO identifier may occur at several
indices; select the occurrence(s) supported by the target patient's sentence. Do not infer that every
phenotype in an article belongs to the target patient. Do not select a candidate merely because it is
near a patient mention: the wording must identify this patient's finding. Exclude findings for relatives,
other cases, controls, population summaries, and explicitly negated findings.
Pay special attention to case numbering, family relationships, pronouns, tables, and comparison sentences.

This is one candidate chunk. A missing candidate is simply not part of this chunk and must not be guessed.
Return exactly this schema, with global candidate indices:
{{"entity_indices":[0,1]}}
"""


def _chunks(values: list[JsonObject], max_chars: int = 12_000) -> Iterable[list[JsonObject]]:
    chunk: list[JsonObject] = []
    chunk_size = 0
    for value in values:
        value_size = len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        if chunk and chunk_size + value_size > max_chars:
            yield chunk
            chunk = []
            chunk_size = 0
        chunk.append(value)
        chunk_size += value_size
    if chunk:
        yield chunk


def associate_with_llm(
    document: JsonObject,
    entities: list[JsonObject],
    client: BigModelClient,
) -> list[JsonObject]:
    positive_entities = [entity for entity in entities if entity.get("note") != "NO"]
    patient_ids = [str(patient["patient_id"]) for patient in document.get("patient", [])]
    if not patient_ids:
        return []
    if not positive_entities:
        return [{"patient_id": patient_id, "phenotype": []} for patient_id in patient_ids]

    selected: dict[str, set[int]] = defaultdict(set)
    indexed_entities = [
        {
            "index": index,
            "passage": _passage_index(document, entity),
            "offset": entity["offset"],
            "length": entity["length"],
            "text": entity["text"],
            "identifier": entity["identifier"],
            "context": _candidate_context(document, entity),
        }
        for index, entity in enumerate(positive_entities)
    ]
    patients_by_id = {str(patient["patient_id"]): patient for patient in document.get("patient", [])}
    for patient_id in patient_ids:
        patient = patients_by_id[patient_id]
        for chunk in _chunks(indexed_entities):
            response = client.chat_json(
                [
                    {"role": "system", "content": _ASSOCIATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _patient_association_prompt(document, patient, chunk),
                    },
                ],
                max_tokens=900,
            )
            values = response.get("entity_indices", []) if isinstance(response, dict) else []
            chunk_indices = {int(item["index"]) for item in chunk}
            for value in values:
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    continue
                if index in chunk_indices:
                    selected[patient_id].add(index)

    associations: list[JsonObject] = []
    for patient_id in patient_ids:
        values: list[str] = []
        seen_values: set[str] = set()
        for index in sorted(selected.get(patient_id, set()), key=lambda item: positive_entities[item]["offset"]):
            value = _entity_value(positive_entities[index])
            if value not in seen_values:
                seen_values.add(value)
                values.append(value)
        associations.append({"patient_id": patient_id, "phenotype": values})
    return associations


def associate_by_proximity(
    document: JsonObject,
    entities: list[JsonObject],
) -> list[JsonObject]:
    """Conservative no-API fallback; the LLM path is preferred for final runs."""
    patients = document.get("patient", [])
    positive_entities = [entity for entity in entities if entity.get("note") != "NO"]
    patient_offsets: dict[str, list[int]] = {
        str(patient["patient_id"]): [int(item["offset"]) for item in patient.get("mention", [])]
        for patient in patients
    }
    selected: dict[str, list[JsonObject]] = defaultdict(list)

    for entity in positive_entities:
        located = passage_for_offset(document, int(entity["offset"]))
        section = located[1].get("section_type") if located else ""
        distances = {
            patient_id: min(
                (abs(int(entity["offset"]) - offset) for offset in offsets),
                default=10**12,
            )
            for patient_id, offsets in patient_offsets.items()
        }
        if not distances:
            continue
        patient_id, distance = min(distances.items(), key=lambda item: item[1])
        if distance <= 1500 and section in {"ABSTRACT", "CASE", "RESULTS", "TABLE", "FIG", "CONCL"}:
            selected[patient_id].append(entity)

    output: list[JsonObject] = []
    for patient in patients:
        patient_id = str(patient["patient_id"])
        values: list[str] = []
        seen: set[str] = set()
        for entity in sorted(selected.get(patient_id, []), key=lambda item: item["offset"]):
            value = _entity_value(entity)
            if value not in seen:
                seen.add(value)
                values.append(value)
        output.append({"patient_id": patient_id, "phenotype": values})
    return output
