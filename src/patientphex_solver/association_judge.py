from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .association import (
    _candidate_context,
    _patient_context,
    _structural_patient_ids,
)
from .io import passage_for_offset
from .llm import BigModelClient
from .ontology import HpoOntology

JsonObject = dict[str, Any]

_SYSTEM_PROMPT = """You reproduce the patient-to-phenotype association policy of the PatientPheX
biomedical benchmark. For one listed patient, decide which supplied phenotype values were explicitly
observed in, diagnosed in, or attributed to that patient. Calibration examples are authoritative
examples from the same dataset.

Do not associate a value merely because it is a disease feature or appears in the article. Exclude
background statements, literature summaries, differential diagnoses, controls, and findings belonging
only to relatives or other patients. Resolve case numbers, pedigree labels, pronouns, and shared findings
from the supplied patient and occurrence contexts. Do not infer phenotypes from a gene or diagnosis.
Return JSON only, using supplied indices; never invent a phenotype value."""


@dataclass(frozen=True, slots=True)
class AssociationCalibrationExample:
    pmc_id: str
    accepted: bool
    value: str
    hpo_name: str
    patient_context: str
    occurrence_context: str


def _entity_values(entity: JsonObject) -> list[str]:
    identifier = str(entity["identifier"])
    if identifier == "-1":
        return [str(entity["text"])]
    return [value for value in identifier.split(";") if value]


def _hpo_name(ontology: HpoOntology, value: str) -> str:
    term = ontology.terms.get(ontology.canonical_id(value))
    return term.name if term is not None and term.name else value


def _patient_summary(document: JsonObject, patient: JsonObject) -> str:
    contexts = _patient_context(document, patient)
    compact = [" ".join(value.split())[:260] for value in contexts[:3]]
    mentions = ", ".join(
        f"{item.get('text')}@{item.get('offset')}"
        for item in patient.get("mention", [])[:6]
    )
    return f"mentions: {mentions}; contexts: {' | '.join(compact)}"


def _occurrence_summary(document: JsonObject, entity: JsonObject) -> str:
    located = passage_for_offset(document, int(entity["offset"]))
    section = str(located[1].get("section_type", "UNKNOWN")) if located else "UNKNOWN"
    return (
        f"{section}@{entity['offset']}: "
        f"{_candidate_context(document, entity)[:300]}"
    )


def _group_entities(entities: Iterable[JsonObject]) -> dict[str, list[JsonObject]]:
    grouped: dict[str, list[JsonObject]] = defaultdict(list)
    for entity in entities:
        if entity.get("note") == "NO":
            continue
        for value in _entity_values(entity):
            grouped[value].append(entity)
    return dict(grouped)


def build_association_calibration_examples(
    training_documents: Iterable[JsonObject],
    ontology: HpoOntology,
) -> list[AssociationCalibrationExample]:
    examples: list[AssociationCalibrationExample] = []
    for document in training_documents:
        entities_by_value = _group_entities(document.get("entities", []))
        gold_by_patient = {
            str(item["patient_id"]): set(item.get("phenotype", []))
            for item in document.get("association", [])
        }
        for patient in document.get("patient", []):
            patient_id = str(patient["patient_id"])
            patient_context = _patient_summary(document, patient)
            gold_values = gold_by_patient.get(patient_id, set())
            for value, entities in entities_by_value.items():
                examples.append(
                    AssociationCalibrationExample(
                        pmc_id=str(document["pmc_id"]),
                        accepted=value in gold_values,
                        value=value,
                        hpo_name=_hpo_name(ontology, value),
                        patient_context=patient_context,
                        occurrence_context=" | ".join(
                            _occurrence_summary(document, entity)
                            for entity in entities[:2]
                        ),
                    )
                )
    return examples


def _stable_tiebreak(example: AssociationCalibrationExample) -> str:
    material = "\x1f".join(
        [
            example.pmc_id,
            example.value,
            example.patient_context,
            example.occurrence_context,
            "1" if example.accepted else "0",
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _select_examples(
    examples: Iterable[AssociationCalibrationExample],
    values: set[str],
    *,
    exclude_pmc_id: str | None,
    per_label: int,
) -> list[AssociationCalibrationExample]:
    source = [
        item for item in examples if item.pmc_id != str(exclude_pmc_id)
    ]

    def rank(item: AssociationCalibrationExample) -> tuple[int, str]:
        return (1 if item.value in values else 0, _stable_tiebreak(item))

    selected: list[AssociationCalibrationExample] = []
    for accepted in (True, False):
        matching = [item for item in source if item.accepted is accepted]
        matching.sort(key=rank, reverse=True)
        selected.extend(matching[:per_label])
    return selected


def _rank_occurrences_for_patient(
    patient: JsonObject,
    entities: list[JsonObject],
) -> list[JsonObject]:
    offsets = [int(item["offset"]) for item in patient.get("mention", [])]

    def distance(entity: JsonObject) -> tuple[int, int]:
        offset = int(entity["offset"])
        nearest = min((abs(offset - value) for value in offsets), default=10**12)
        return nearest, offset

    return sorted(entities, key=distance)


def _candidate_rows(
    document: JsonObject,
    patient: JsonObject,
    grouped: dict[str, list[JsonObject]],
    values: list[str],
    ontology: HpoOntology,
) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for index, value in enumerate(values):
        occurrences = _rank_occurrences_for_patient(patient, grouped[value])[:3]
        rows.append(
            {
                "index": index,
                "value": value,
                "hpo_name": _hpo_name(ontology, value),
                "occurrences": [
                    _occurrence_summary(document, entity) for entity in occurrences
                ],
            }
        )
    return rows


def _joint_candidate_rows(
    document: JsonObject,
    grouped: dict[str, list[JsonObject]],
    values: list[str],
    ontology: HpoOntology,
) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for index, value in enumerate(values):
        rows.append(
            {
                "index": index,
                "value": value,
                "hpo_name": _hpo_name(ontology, value),
                "occurrences": [
                    _occurrence_summary(document, entity)
                    for entity in grouped[value][:5]
                ],
            }
        )
    return rows


def _prompt(
    document: JsonObject,
    patient: JsonObject,
    rows: list[JsonObject],
    examples: list[AssociationCalibrationExample],
) -> str:
    calibration = [
        {
            "label": "ASSOCIATE" if item.accepted else "DO_NOT_ASSOCIATE",
            "value": item.value,
            "hpo_name": item.hpo_name,
            "patient_context": item.patient_context,
            "occurrence_context": item.occurrence_context,
        }
        for item in examples
    ]
    return f"""PMC_ID: {document['pmc_id']}

TARGET PATIENT:
{json.dumps({
    'patient_id': str(patient['patient_id']),
    'summary': _patient_summary(document, patient),
}, ensure_ascii=False, separators=(',', ':'))}

CALIBRATION EXAMPLES FROM OTHER ARTICLES:
{json.dumps(calibration, ensure_ascii=False, separators=(',', ':'))}

CANDIDATE PHENOTYPE VALUES:
{json.dumps(rows, ensure_ascii=False, separators=(',', ':'))}

Return exactly this schema:
{{"associated_indices":[0,2],"uncertain_indices":[5]}}
Only clear, explicit patient associations belong in associated_indices. Borderline values belong only
in uncertain_indices. An occurrence may support the value even when another occurrence is background.
"""


def _joint_prompt(
    document: JsonObject,
    patients: list[JsonObject],
    rows: list[JsonObject],
    examples: list[AssociationCalibrationExample],
) -> str:
    calibration = [
        {
            "label": "ASSOCIATE" if item.accepted else "DO_NOT_ASSOCIATE",
            "value": item.value,
            "hpo_name": item.hpo_name,
            "patient_context": item.patient_context,
            "occurrence_context": item.occurrence_context,
        }
        for item in examples
    ]
    patient_rows = [
        {
            "patient_id": str(patient["patient_id"]),
            "summary": _patient_summary(document, patient),
        }
        for patient in patients
    ]
    return f"""PMC_ID: {document['pmc_id']}

LISTED PATIENTS:
{json.dumps(patient_rows, ensure_ascii=False, separators=(',', ':'))}

CALIBRATION EXAMPLES FROM OTHER ARTICLES:
{json.dumps(calibration, ensure_ascii=False, separators=(',', ':'))}

CANDIDATE PHENOTYPE VALUES AND ALL AVAILABLE OCCURRENCES:
{json.dumps(rows, ensure_ascii=False, separators=(',', ':'))}

Assign each candidate value to every listed patient for whom at least one supplied occurrence is
explicitly attributable. Patients compete for occurrences: do not copy a value to all patients merely
because it is a general disease feature. A value may be assigned to multiple patients only when the
source explicitly supports sharing, such as both twins or all affected siblings. Exclude relatives not
listed as patients, controls, background, differential diagnoses, and negated findings.

Return exactly:
{{"assignments":{{"P1":[0,2],"P2":[]}},"uncertain":{{"P1":[5]}}}}
Use every listed patient ID as a key in assignments. Put borderline indices only in uncertain. Never
invent a value or index.
"""


def _indices(response: Any, key: str, size: int) -> set[int]:
    if not isinstance(response, dict) or not isinstance(response.get(key), list):
        return set()
    selected: set[int] = set()
    for value in response[key]:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < size:
            selected.add(index)
    return selected


def _assignment_indices(response: Any, patient_id: str, key: str, size: int) -> set[int]:
    if not isinstance(response, dict):
        return set()
    assignments = response.get(key, {})
    if not isinstance(assignments, dict):
        return set()
    values = assignments.get(patient_id, [])
    if not isinstance(values, list):
        return set()
    selected: set[int] = set()
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < size:
            selected.add(index)
    return selected


def associate_values_calibrated_with_llm(
    document: JsonObject,
    entities: list[JsonObject],
    ontology: HpoOntology,
    client: BigModelClient,
    calibration_examples: list[AssociationCalibrationExample],
    *,
    batch_size: int = 30,
    calibration_per_label: int = 8,
    include_uncertain: bool = False,
    exclude_calibration_pmc_id: str | None = None,
    structure_multi_patient: bool = True,
    previous_distance: int = 1500,
    next_distance: int = 300,
) -> list[JsonObject]:
    grouped = _group_entities(entities)
    all_values = sorted(grouped, key=lambda value: int(grouped[value][0]["offset"]))
    structural_support: dict[str, set[str]] = defaultdict(set)
    if structure_multi_patient and len(document.get("patient", [])) > 1:
        for value, occurrences in grouped.items():
            for entity in occurrences:
                structural_support[value].update(
                    _structural_patient_ids(
                        document,
                        entity,
                        previous_distance=previous_distance,
                        next_distance=next_distance,
                    )
                )
    associations: list[JsonObject] = []
    for patient in document.get("patient", []):
        accepted: set[str] = set()
        for start in range(0, len(all_values), batch_size):
            values = all_values[start : start + batch_size]
            rows = _candidate_rows(document, patient, grouped, values, ontology)
            examples = _select_examples(
                calibration_examples,
                set(values),
                exclude_pmc_id=exclude_calibration_pmc_id,
                per_label=calibration_per_label,
            )
            response = client.chat_json(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _prompt(document, patient, rows, examples),
                    },
                ],
                max_tokens=1200,
            )
            indices = _indices(response, "associated_indices", len(values))
            if include_uncertain:
                indices.update(_indices(response, "uncertain_indices", len(values)))
            accepted.update(values[index] for index in indices)
        if structural_support:
            patient_id = str(patient["patient_id"])
            accepted = {
                value
                for value in accepted
                if patient_id in structural_support.get(value, set())
            }
        associations.append(
            {
                "patient_id": str(patient["patient_id"]),
                "phenotype": [value for value in all_values if value in accepted],
            }
        )
    return associations


def associate_values_joint_calibrated_with_llm(
    document: JsonObject,
    entities: list[JsonObject],
    ontology: HpoOntology,
    client: BigModelClient,
    calibration_examples: list[AssociationCalibrationExample],
    *,
    batch_size: int = 25,
    calibration_per_label: int = 8,
    include_uncertain: bool = False,
    exclude_calibration_pmc_id: str | None = None,
    structure_multi_patient: bool = True,
    previous_distance: int = 1500,
    next_distance: int = 300,
) -> list[JsonObject]:
    """Assign unique phenotype values jointly while preserving occurrence evidence."""
    patients = list(document.get("patient", []))
    patient_ids = [str(patient["patient_id"]) for patient in patients]
    grouped = _group_entities(entities)
    all_values = sorted(grouped, key=lambda value: int(grouped[value][0]["offset"]))
    structural_support: dict[str, set[str]] = defaultdict(set)
    if structure_multi_patient and len(patients) > 1:
        for value, occurrences in grouped.items():
            for entity in occurrences:
                structural_support[value].update(
                    _structural_patient_ids(
                        document,
                        entity,
                        previous_distance=previous_distance,
                        next_distance=next_distance,
                    )
                )
    selected: dict[str, set[str]] = defaultdict(set)
    for start in range(0, len(all_values), batch_size):
        values = all_values[start : start + batch_size]
        rows = _joint_candidate_rows(document, grouped, values, ontology)
        examples = _select_examples(
            calibration_examples,
            set(values),
            exclude_pmc_id=exclude_calibration_pmc_id,
            per_label=calibration_per_label,
        )
        try:
            response = client.chat_json(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _joint_prompt(document, patients, rows, examples),
                    },
                ],
                max_tokens=1500,
            )
        except (json.JSONDecodeError, RuntimeError):
            # One malformed or failed batch must not discard earlier batches.
            continue
        for patient_id in patient_ids:
            indices = _assignment_indices(
                response, patient_id, "assignments", len(values)
            )
            if include_uncertain:
                indices.update(
                    _assignment_indices(response, patient_id, "uncertain", len(values))
                )
            selected[patient_id].update(values[index] for index in indices)
    associations: list[JsonObject] = []
    for patient_id in patient_ids:
        values = selected.get(patient_id, set())
        if structural_support:
            values = {
                value
                for value in values
                if patient_id in structural_support.get(value, set())
            }
        associations.append(
            {
                "patient_id": patient_id,
                "phenotype": [value for value in all_values if value in values],
            }
        )
    return associations
