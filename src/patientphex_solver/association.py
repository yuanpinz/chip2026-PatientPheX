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
Associate phenotype entity candidates with the specific patients described in a PMC article,
reasoning over all patients jointly.
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


def _structural_patient_ids(
    document: JsonObject,
    entity: JsonObject,
    *,
    previous_distance: int = 3000,
    next_distance: int = 300,
) -> set[str]:
    """Return patients whose local article structure can support an entity.

    Patient mentions in the same sentence are strongest.  If a passage contains
    a patient heading followed by several finding sentences, all patients named
    in that passage remain possible.  For passages without a local mention, a
    short continuation window is used in both directions so headings and table
    captions do not break a case block.
    """
    located = passage_for_offset(document, int(entity["offset"]))
    if located is None:
        return set()
    passage_index, passage = located
    entity_offset = int(entity["offset"])
    passage_offset = int(passage["offset"])
    entity_start = entity_offset - passage_offset
    entity_end = entity_start + int(entity["length"])
    passage_text = str(passage.get("text", ""))
    sentence_start, sentence_end = _sentence_bounds(
        passage_text, entity_start, entity_end
    )

    anchors: list[tuple[int, str, int]] = []
    for patient in document.get("patient", []):
        patient_id = str(patient["patient_id"])
        for mention in patient.get("mention", []):
            mention_offset = int(mention["offset"])
            mention_located = passage_for_offset(document, mention_offset)
            anchors.append(
                (
                    mention_offset,
                    patient_id,
                    mention_located[0] if mention_located is not None else -1,
                )
            )
    if not anchors:
        return set()

    same_passage = [item for item in anchors if item[2] == passage_index]
    same_sentence = {
        patient_id
        for mention_offset, patient_id, _ in same_passage
        if sentence_start
        <= mention_offset - passage_offset
        < sentence_end
    }
    if same_sentence:
        return same_sentence
    if same_passage:
        return {patient_id for _, patient_id, _ in same_passage}

    previous = [item for item in anchors if item[0] <= entity_offset]
    following = [item for item in anchors if item[0] > entity_offset]
    possible: set[str] = set()
    if previous and entity_offset - previous[-1][0] <= previous_distance:
        possible.add(previous[-1][1])
    if following and following[0][0] - entity_offset <= next_distance:
        possible.add(following[0][1])
    return possible


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


def _joint_patient_context(document: JsonObject) -> list[dict[str, Any]]:
    patients = []
    for patient in document.get("patient", []):
        patients.append(
            {
                "patient_id": str(patient["patient_id"]),
                "mentions": patient.get("mention", []),
                "mention_context": _patient_context(document, patient),
            }
        )
    return patients


def _joint_association_prompt(
    document: JsonObject,
    patients: list[JsonObject],
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

PATIENTS:
{json.dumps(_joint_patient_context(document), ensure_ascii=False)}

POSITIVE PHENOTYPE ENTITY CANDIDATES:
{json.dumps(entity_rows, ensure_ascii=False)}

For every candidate, decide which listed patient(s), if any, the finding belongs to.
Use the wording in the candidate's sentence and the article structure. A candidate may be assigned
to more than one patient only when the sentence explicitly says the finding is shared by them, for
example "both twins" or "all patients". Do not copy a phenotype to every patient merely because it
is a general feature of the disease. Findings about relatives, controls, unaffected people,
population summaries, differential diagnoses, and explicitly negated findings must be omitted.
When a sentence names one patient, case, proband, twin, or pedigree individual, assign it only there.
If the candidate is not attributable to a listed patient, omit it.

Return exactly this schema, using the supplied global candidate indices and every listed patient ID:
{{"assignments":{{"PATIENT_ID":[0,1]}}}}
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


def _joint_entity_indices(
    document: JsonObject,
    entities: list[JsonObject],
    client: BigModelClient,
) -> tuple[list[JsonObject], dict[str, set[int]]]:
    patients = document.get("patient", [])
    patient_ids = [str(patient["patient_id"]) for patient in patients]
    positive_entities = [entity for entity in entities if entity.get("note") != "NO"]
    if not patient_ids or not positive_entities:
        return positive_entities, defaultdict(set)

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
    selected: dict[str, set[int]] = defaultdict(set)
    valid_patient_ids = set(patient_ids)
    for chunk in _chunks(indexed_entities, max_chars=14_000):
        try:
            response = client.chat_json(
                [
                    {"role": "system", "content": _ASSOCIATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _joint_association_prompt(document, patients, chunk),
                    },
                ],
                max_tokens=1400,
            )
        except (json.JSONDecodeError, RuntimeError):
            continue
        assignments = response.get("assignments", {}) if isinstance(response, dict) else {}
        if not isinstance(assignments, dict):
            continue
        chunk_indices = {int(item["index"]) for item in chunk}
        for patient_id, values in assignments.items():
            patient_id = str(patient_id)
            if patient_id not in valid_patient_ids or not isinstance(values, list):
                continue
            for value in values:
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    continue
                if index in chunk_indices:
                    selected[patient_id].add(index)
    return positive_entities, selected


def _associations_from_indices(
    patient_ids: list[str],
    positive_entities: list[JsonObject],
    selected: dict[str, set[int]],
) -> list[JsonObject]:
    associations: list[JsonObject] = []
    for patient_id in patient_ids:
        values: list[str] = []
        seen_values: set[str] = set()
        for index in sorted(
            selected.get(patient_id, set()),
            key=lambda item: int(positive_entities[item]["offset"]),
        ):
            value = _entity_value(positive_entities[index])
            if value not in seen_values:
                seen_values.add(value)
                values.append(value)
        associations.append({"patient_id": patient_id, "phenotype": values})
    return associations


def associate_with_llm(
    document: JsonObject,
    entities: list[JsonObject],
    client: BigModelClient,
) -> list[JsonObject]:
    positive_entities, selected = _patient_entity_indices(document, entities, client)
    patient_ids = [str(patient["patient_id"]) for patient in document.get("patient", [])]
    if not positive_entities:
        return [{"patient_id": patient_id, "phenotype": []} for patient_id in patient_ids]
    return _associations_from_indices(patient_ids, positive_entities, selected)


def _patient_entity_indices(
    document: JsonObject,
    entities: list[JsonObject],
    client: BigModelClient,
) -> tuple[list[JsonObject], dict[str, set[int]]]:
    """Select entity occurrences independently for each listed patient."""
    positive_entities = [entity for entity in entities if entity.get("note") != "NO"]
    patient_ids = [str(patient["patient_id"]) for patient in document.get("patient", [])]
    if not patient_ids:
        return positive_entities, defaultdict(set)
    if not positive_entities:
        return positive_entities, defaultdict(set)

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
            try:
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
            except (json.JSONDecodeError, RuntimeError):
                continue
            values = response.get("entity_indices", []) if isinstance(response, dict) else []
            chunk_indices = {int(item["index"]) for item in chunk}
            for value in values:
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    continue
                if index in chunk_indices:
                    selected[patient_id].add(index)

    return positive_entities, selected


def associate_patient_structured_with_llm(
    document: JsonObject,
    entities: list[JsonObject],
    client: BigModelClient,
) -> list[JsonObject]:
    """Use per-patient LLM selections constrained at occurrence level."""
    patient_ids = [str(patient["patient_id"]) for patient in document.get("patient", [])]
    positive_entities, selected = _patient_entity_indices(document, entities, client)
    if not positive_entities:
        return [{"patient_id": patient_id, "phenotype": []} for patient_id in patient_ids]
    selected = _filter_selected_indices_by_structure(
        document,
        positive_entities,
        selected,
        previous_distance=3000,
        next_distance=500,
    )
    return _associations_from_indices(patient_ids, positive_entities, selected)


def associate_joint_with_llm(
    document: JsonObject,
    entities: list[JsonObject],
    client: BigModelClient,
) -> list[JsonObject]:
    """Assign candidate occurrences jointly so patients compete for each finding."""
    patients = document.get("patient", [])
    patient_ids = [str(patient["patient_id"]) for patient in patients]
    if not patient_ids:
        return []
    positive_entities, selected = _joint_entity_indices(document, entities, client)
    if not positive_entities:
        return [{"patient_id": patient_id, "phenotype": []} for patient_id in patient_ids]

    return _associations_from_indices(patient_ids, positive_entities, selected)


def filter_associations_by_structure(
    document: JsonObject,
    entities: list[JsonObject],
    associations: list[JsonObject],
    *,
    previous_distance: int = 3000,
    next_distance: int = 300,
) -> list[JsonObject]:
    """Remove multi-patient assignments unsupported by local article structure."""
    patients = document.get("patient", [])
    if len(patients) <= 1:
        return associations

    value_patients: dict[str, set[str]] = defaultdict(set)
    for entity in entities:
        if entity.get("note") == "NO":
            continue
        value = _entity_value(entity)
        value_patients[value].update(
            _structural_patient_ids(
                document,
                entity,
                previous_distance=previous_distance,
                next_distance=next_distance,
            )
        )

    filtered: list[JsonObject] = []
    for association in associations:
        patient_id = str(association.get("patient_id"))
        phenotype = [
            value
            for value in association.get("phenotype", [])
            if patient_id in value_patients.get(str(value), set())
        ]
        filtered.append({**association, "phenotype": phenotype})
    return filtered


def _filter_selected_indices_by_structure(
    document: JsonObject,
    positive_entities: list[JsonObject],
    selected: dict[str, set[int]],
    *,
    previous_distance: int = 3000,
    next_distance: int = 300,
) -> dict[str, set[int]]:
    filtered: dict[str, set[int]] = defaultdict(set)
    for patient_id, indices in selected.items():
        for index in indices:
            possible = _structural_patient_ids(
                document,
                positive_entities[index],
                previous_distance=previous_distance,
                next_distance=next_distance,
            )
            if patient_id in possible:
                filtered[patient_id].add(index)
    return filtered


def associate_joint_structured_with_llm(
    document: JsonObject,
    entities: list[JsonObject],
    client: BigModelClient,
) -> list[JsonObject]:
    """Use joint LLM assignments constrained by patient-local article structure."""
    patient_ids = [str(patient["patient_id"]) for patient in document.get("patient", [])]
    positive_entities, selected = _joint_entity_indices(document, entities, client)
    if not positive_entities:
        return [{"patient_id": patient_id, "phenotype": []} for patient_id in patient_ids]
    selected = _filter_selected_indices_by_structure(
        document,
        positive_entities,
        selected,
    )
    return _associations_from_indices(patient_ids, positive_entities, selected)


def associate_by_proximity(
    document: JsonObject,
    entities: list[JsonObject],
    *,
    same_passage_distance: int = 3500,
    global_distance: int = 1200,
) -> list[JsonObject]:
    """Assign findings to the nearest patient, preferring the containing passage."""
    patients = document.get("patient", [])
    positive_entities = [entity for entity in entities if entity.get("note") != "NO"]
    patient_offsets: dict[str, list[int]] = {
        str(patient["patient_id"]): [int(item["offset"]) for item in patient.get("mention", [])]
        for patient in patients
    }
    passage_patient_offsets: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for patient_id, offsets in patient_offsets.items():
        for offset in offsets:
            located = passage_for_offset(document, offset)
            if located is not None:
                passage_patient_offsets[located[0]][patient_id].append(offset)
    selected: dict[str, list[JsonObject]] = defaultdict(list)

    for entity in positive_entities:
        located = passage_for_offset(document, int(entity["offset"]))
        if located is None or not patient_offsets:
            continue
        entity_offset = int(entity["offset"])
        passage_index = located[0]
        local_patients = passage_patient_offsets.get(passage_index, {})
        if local_patients:
            distances = {
                patient_id: min(abs(entity_offset - offset) for offset in offsets)
                for patient_id, offsets in local_patients.items()
            }
            patient_id, distance = min(distances.items(), key=lambda item: (item[1], item[0]))
            if distance <= same_passage_distance:
                selected[patient_id].append(entity)
            continue

        distances = {
            patient_id: min(
                (abs(entity_offset - offset) for offset in offsets),
                default=10**12,
            )
            for patient_id, offsets in patient_offsets.items()
        }
        patient_id, distance = min(distances.items(), key=lambda item: (item[1], item[0]))
        if len(patient_offsets) == 1 or distance <= global_distance:
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
