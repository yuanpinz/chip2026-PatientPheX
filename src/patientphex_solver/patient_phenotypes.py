from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from rapidfuzz import fuzz

from .entities import GazetteerExtractor, merge_entities
from .llm import BigModelClient
from .llm_entities import _aligned_start
from .ontology import HpoOntology, normalize_surface

JsonObject = dict[str, Any]

_PATIENT_PHENOTYPE_SYSTEM_PROMPT = """You are an expert annotator for a patient phenotype benchmark.
Extract every phenotype explicitly observed in or diagnosed in each listed patient from a complete
biomedical case article. Include symptoms, diagnoses, malformations, abnormal measurements,
developmental and behavioral findings, complications, and past findings. Exclude general background
statements, findings only attributed to relatives or other unlisted people, differential diagnoses,
normal findings, and explicitly absent or negated findings. Do not infer findings from a gene or disease.

Each result must quote the shortest exact source span that expresses the finding, with its passage index
and passage-local zero-based start. Give the closest canonical Human Phenotype Ontology phrase and HPO
identifier when known. Never invent source text. Return JSON only."""


def _passage_chunks(
    document: JsonObject, max_chars: int | None
) -> list[list[tuple[int, JsonObject]]]:
    indexed = list(enumerate(document.get("full_text", [])))
    if max_chars is None:
        return [indexed]
    chunks: list[list[tuple[int, JsonObject]]] = []
    chunk: list[tuple[int, JsonObject]] = []
    size = 0
    for item in indexed:
        passage_size = len(str(item[1].get("text", ""))) + 100
        if chunk and size + passage_size > max_chars:
            chunks.append(chunk)
            chunk = []
            size = 0
        chunk.append(item)
        size += passage_size
    if chunk:
        chunks.append(chunk)
    return chunks


def _patient_phenotype_prompt(
    document: JsonObject,
    indexed_passages: list[tuple[int, JsonObject]] | None = None,
) -> str:
    patients = [
        {
            "patient_id": str(patient["patient_id"]),
            "mentions": patient.get("mention", []),
        }
        for patient in document.get("patient", [])
    ]
    passages = [
        {
            "passage_index": index,
            "section": passage.get("section_type"),
            "text": passage.get("text", ""),
        }
        for index, passage in (
            indexed_passages
            if indexed_passages is not None
            else enumerate(document.get("full_text", []))
        )
    ]
    return f"""PMC_ID: {document["pmc_id"]}

LISTED PATIENTS:
{json.dumps(patients, ensure_ascii=False)}

ARTICLE PASSAGE CHUNK:
{json.dumps(passages, ensure_ascii=False)}

Return exactly this schema:
{{"assignments":{{"P1":[{{"passage_index":3,"start":42,"text":"exact source span","canonical":"canonical HPO phrase","hpo_id":"HP:0000000"}}]}}}}

This is one passage chunk from the article. Use every listed patient_id as a key, even when its list is
empty. Repeated evidence for the same HPO
concept is unnecessary within one patient. When a sentence lists several distinct findings, return one
row per finding. A result is invalid unless ARTICLE PASSAGE CHUNK[passage_index].text[start:start+len(text)]
equals text exactly. Do not return benchmark entity indices or explanatory prose.
"""


def _training_identifier(extractor: GazetteerExtractor, value: str) -> str | None:
    alias = normalize_surface(value)
    stats = extractor.surface_stats.get(alias)
    if stats is None or not stats.positive_identifier_counts:
        return None
    identifier, count = stats.positive_identifier_counts.most_common(1)[0]
    if count * 2 < sum(stats.positive_identifier_counts.values()):
        return None
    return identifier if identifier != "-1" else None


def _identifier_for_finding(
    row: JsonObject,
    ontology: HpoOntology,
    extractor: GazetteerExtractor,
) -> str | None:
    text = str(row.get("text", "")).strip()
    canonical = str(row.get("canonical", "")).strip()

    # The model's explicit HPO ID carries more disambiguating information than
    # a surface-form lookup. Accept it only when it is a known non-root term and
    # its supplied phrase is reasonably compatible with that term.
    supplied = ontology.canonical_id(str(row.get("hpo_id", "")).strip())
    if supplied in ontology.descendants and supplied != ontology.root:
        term = ontology.terms.get(supplied)
        aliases = [term.name, *term.synonyms] if term is not None else []
        evidence = [value for value in (canonical, text) if value]
        if not aliases:
            return supplied
        if any(
            max(
                (
                    fuzz.WRatio(normalize_surface(value), normalize_surface(alias))
                    for alias in aliases
                ),
                default=0.0,
            )
            >= 72.0
            for value in evidence
        ):
            return supplied

    for value in (text, canonical):
        if not value:
            continue
        identifier = _training_identifier(extractor, value) or ontology.resolve_alias(
            value, id_frequency=extractor.id_frequency
        )
        if identifier:
            return identifier

    link = ontology.link(
        [canonical, text],
        id_frequency=extractor.id_frequency,
        score_cutoff=88.0,
    )
    return link.identifier if link is not None else None


def _response_assignments(response: Any) -> dict[str, list[JsonObject]]:
    if not isinstance(response, dict):
        return {}
    assignments = response.get("assignments", {})
    if not isinstance(assignments, dict):
        return {}
    return {
        str(patient_id): [row for row in rows if isinstance(row, dict)]
        for patient_id, rows in assignments.items()
        if isinstance(rows, list)
    }


def discover_patient_phenotypes_with_llm(
    document: JsonObject,
    ontology: HpoOntology,
    extractor: GazetteerExtractor,
    client: BigModelClient,
    *,
    max_tokens: int = 6000,
    passage_chunk_chars: int | None = None,
    fallback_chunk_chars: int = 9000,
) -> tuple[list[JsonObject], list[JsonObject]]:
    """Extract patient-attributed findings without limiting recall to gazetteer candidates."""
    assignments: dict[str, list[JsonObject]] = defaultdict(list)
    successful_chunks = 0
    last_error: Exception | None = None
    chunk_plans = [_passage_chunks(document, passage_chunk_chars)]
    if passage_chunk_chars is None and fallback_chunk_chars > 0:
        chunk_plans.append(_passage_chunks(document, fallback_chunk_chars))

    for chunks in chunk_plans:
        for indexed_passages in chunks:
            try:
                response = client.chat_json(
                    [
                        {"role": "system", "content": _PATIENT_PHENOTYPE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _patient_phenotype_prompt(
                                document, indexed_passages
                            ),
                        },
                    ],
                    max_tokens=max_tokens,
                )
            except (json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                continue
            successful_chunks += 1
            for patient_id, rows in _response_assignments(response).items():
                assignments[patient_id].extend(rows)
        if successful_chunks:
            break
    if successful_chunks == 0 and last_error is not None:
        raise RuntimeError("all direct phenotype chunks failed") from last_error
    patient_ids = [
        str(patient["patient_id"]) for patient in document.get("patient", [])
    ]
    valid_patient_ids = set(patient_ids)
    passages = document.get("full_text", [])
    additions: list[JsonObject] = []
    values_by_patient: dict[str, list[str]] = defaultdict(list)
    seen_by_patient: dict[str, set[str]] = defaultdict(set)

    for patient_id, rows in assignments.items():
        if patient_id not in valid_patient_ids:
            continue
        for row in rows:
            try:
                passage_index = int(row["passage_index"])
                declared_start = int(row["start"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= passage_index < len(passages):
                continue
            passage = passages[passage_index]
            passage_text = str(passage.get("text", ""))
            text = str(row.get("text", ""))
            start = _aligned_start(passage_text, declared_start, text)
            if start is None or not text.strip():
                continue
            identifier = _identifier_for_finding(row, ontology, extractor)
            if identifier is None:
                continue
            offset = int(passage["offset"]) + start
            additions.append(
                {
                    "identifier": identifier,
                    "type": "Phenotype",
                    "offset": offset,
                    "length": len(text),
                    "text": text,
                    "note": None,
                }
            )
            for value in identifier.split(";"):
                if value not in seen_by_patient[patient_id]:
                    seen_by_patient[patient_id].add(value)
                    values_by_patient[patient_id].append(value)

    associations = [
        {"patient_id": patient_id, "phenotype": values_by_patient[patient_id]}
        for patient_id in patient_ids
    ]
    return merge_entities(additions), associations
