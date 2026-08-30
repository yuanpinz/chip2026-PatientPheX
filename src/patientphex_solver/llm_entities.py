from __future__ import annotations

import json
from typing import Any

from .entities import merge_entities
from .llm import BigModelClient
from .ontology import HpoOntology

JsonObject = dict[str, Any]

_ENTITY_SYSTEM_PROMPT = """You extract Human Phenotype Ontology phenotype mentions from biomedical full text.
Find explicit phenotype spans missed by a supplied gazetteer. A phenotype is an abnormal clinical finding,
symptom, malformation, abnormal measurement, developmental/behavioral abnormality, or named phenotype.
Do not extract diseases merely as diagnoses unless they are themselves an HPO phenotype, genes, variants,
treatments, procedures, normal anatomy, or vague words without an abnormal finding.
Copy exact text spans and exact zero-based starts inside each passage. Return JSON only.
Do not output an HPO identifier; give a short canonical English phenotype phrase for local ontology linking."""

_ARTICLE_ENTITY_SYSTEM_PROMPT = """You are an expert annotator for a biomedical phenotype benchmark.
Read the complete PMC article and extract every explicit span that can map to a Human Phenotype
Ontology term in the Phenotypic abnormality branch. The benchmark labels repeated mentions across
title, abstract, introduction, case, figures, results, and discussion, including general/background
disease manifestations. Include diagnoses, symptoms, abnormal measurements, developmental findings,
morphology, and clinical conditions when an HPO term exists. Exclude genes, variants, treatments,
procedures, tests/instruments, normal anatomy, and clearly normal or unaffected findings. Keep an
explicitly negated phenotype and set negated=true. Copy exact text and passage-local zero-based
starts; never invent text. Return JSON only and do not output spans in already_found."""


def _entity_prompt(document: JsonObject, known_entities: list[JsonObject]) -> str:
    return f"""PMC_ID: {document['pmc_id']}

The caller sends one passage chunk at a time. The chunk uses a local zero-based start.

PASSAGE CHUNK AND ALREADY FOUND SPANS:
{{chunk}}

Return only additional phenotype spans not already_found, using this exact schema:
{{"entities":[{{"passage_index":0,"start":12,"text":"exact text","canonical":"canonical phenotype phrase","negated":false}}]}}
If nothing is missing, return {{"entities":[]}}.
"""

_FEWSHOT_ENTITY_SYSTEM_PROMPT = """You annotate phenotype entities for the PatientPheX benchmark.
The examples below are gold annotations from other articles in this same benchmark and define the
annotation policy. Extract every explicit phenotype mention in the target passage chunk that follows
the examples. This includes abnormal findings, symptoms, malformations, abnormal measurements,
developmental/behavioral findings, and disease-like phenotype mentions when the benchmark examples
annotate them. Include repeated and nested mentions when the wording supports them. Keep explicitly
negated findings with negated=true. Exclude genes, variants, treatments, procedures, tests, normal
anatomy, family history without an abnormal finding, and vague non-phenotype text.

Copy the shortest exact source span used by the annotation style. Do not paraphrase, repair grammar,
or include punctuation unless it is part of the annotated span. Return JSON only. Every start is
zero-based relative to its passage. Use a valid HPO ID when the text maps to one; use "-1" when the
gold-style example shows a phenotype span that has no HPO mapping. Never invent text or spans."""


def discover_entities_with_llm(
    document: JsonObject,
    known_entities: list[JsonObject],
    ontology: HpoOntology,
    client: BigModelClient,
    *,
    id_frequency: dict[str, int] | None = None,
    fuzzy_cutoff: float = 82.0,
) -> list[JsonObject]:
    passages = document.get("full_text", [])
    known_spans = {(int(item["offset"]), int(item["length"])) for item in known_entities}
    additions: list[JsonObject] = []
    for passage_index, passage in enumerate(passages):
        passage_text = str(passage.get("text", ""))
        if not passage_text.strip():
            continue
        chunk = {
            "passage_index": passage_index,
            "section": passage["section_type"],
            "text": passage_text,
            "already_found": [
                {
                    "start": int(item["offset"]) - int(passage["offset"]),
                    "text": item["text"],
                }
                for item in known_entities
                if int(passage["offset"]) <= int(item["offset"]) < int(passage["offset"]) + len(passage_text)
            ],
        }
        prompt = _entity_prompt(document, known_entities).replace(
            "{chunk}", json.dumps(chunk, ensure_ascii=False)
        )
        try:
            response = client.chat_json(
                [
                    {"role": "system", "content": _ENTITY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2200,
            )
        except (json.JSONDecodeError, RuntimeError):
            # A single malformed model response must not discard the rest of
            # the article; the gazetteer remains a valid baseline for the
            # affected passage.
            continue
        rows = response.get("entities", []) if isinstance(response, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                local_start = int(row["start"])
            except (KeyError, TypeError, ValueError):
                continue
            text = str(row.get("text", ""))
            if not text or not 0 <= local_start <= len(passage_text) - len(text):
                continue
            if passage_text[local_start : local_start + len(text)] != text:
                continue
            offset = int(passage["offset"]) + local_start
            if (offset, len(text)) in known_spans:
                continue
            link = ontology.link(
                [str(row.get("canonical", "")), text],
                id_frequency=id_frequency,
                score_cutoff=fuzzy_cutoff,
            )
            if link is None:
                continue
            additions.append(
                {
                    "identifier": link.identifier,
                    "type": "Phenotype",
                    "offset": offset,
                    "length": len(text),
                    "text": text,
                    "note": "NO" if row.get("negated") is True else None,
                }
            )
    return merge_entities(additions)


def _article_entity_prompt(
    document: JsonObject,
    known_entities: list[JsonObject],
) -> str:
    passages = [
        {
            "passage_index": index,
            "section": passage.get("section_type"),
            "text": passage.get("text", ""),
        }
        for index, passage in enumerate(document.get("full_text", []))
    ]
    already_found = []
    for entity in known_entities:
        passage_index = _passage_index(document, entity)
        if passage_index is None:
            continue
        already_found.append(
            {
                "passage_index": passage_index,
                "start": int(entity["offset"])
                - int(document["full_text"][passage_index]["offset"]),
                "length": int(entity["length"]),
                "text": entity["text"],
            }
        )
    return f"""PMC_ID: {document['pmc_id']}

ARTICLE PASSAGES:
{json.dumps(passages, ensure_ascii=False)}

ALREADY_FOUND:
{json.dumps(already_found, ensure_ascii=False)}

Return only additional phenotype spans using this exact schema:
{{"entities":[{{"passage_index":0,"start":12,"text":"exact text","canonical":"canonical phenotype phrase","negated":false}}]}}
Use the shortest exact span that expresses the HPO finding. Include repeated occurrences and valid
nested mentions. A row is valid only when its text occurs at the supplied start in that passage.
The source text itself must be a standard HPO term or synonym. Do not output a longer measurement,
description, or clause merely because it can be normalized to an HPO concept; output its shortest
HPO phrase instead. This exact-span constraint is essential for the benchmark.
"""


def _passage_index(document: JsonObject, entity: JsonObject) -> int | None:
    offset = int(entity["offset"])
    for index, passage in enumerate(document.get("full_text", [])):
        start = int(passage["offset"])
        if start <= offset < start + len(str(passage.get("text", ""))):
            return index
    return None


def _aligned_start(text: str, declared_start: int, value: str) -> int | None:
    if 0 <= declared_start <= len(text) - len(value) and text[
        declared_start : declared_start + len(value)
    ] == value:
        return declared_start
    starts: list[int] = []
    cursor = text.find(value)
    while cursor >= 0:
        starts.append(cursor)
        cursor = text.find(value, cursor + 1)
    if not starts:
        return None
    return min(starts, key=lambda start: abs(start - declared_start))


def discover_entities_article_with_llm(
    document: JsonObject,
    known_entities: list[JsonObject],
    ontology: HpoOntology,
    client: BigModelClient,
    *,
    id_frequency: dict[str, int] | None = None,
    fuzzy_cutoff: float = 78.0,
    exact_text_only: bool = True,
) -> list[JsonObject]:
    """Use one article-level call to recover phenotype spans missed by the gazetteer.

    Article-level discovery is deliberately conservative by default: a model-generated
    canonical phrase is not enough to establish an exact benchmark ID for a free-text span.
    """
    try:
        response = client.chat_json(
            [
                {"role": "system", "content": _ARTICLE_ENTITY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _article_entity_prompt(document, known_entities),
                },
            ],
            max_tokens=12000,
        )
    except (json.JSONDecodeError, RuntimeError):
        return []

    known_spans = {
        (int(entity["offset"]), int(entity["length"])) for entity in known_entities
    }
    additions: list[JsonObject] = []
    rows = response.get("entities", []) if isinstance(response, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            passage_index = int(row["passage_index"])
            declared_start = int(row["start"])
        except (KeyError, TypeError, ValueError):
            continue
        passages = document.get("full_text", [])
        if not 0 <= passage_index < len(passages):
            continue
        passage = passages[passage_index]
        passage_text = str(passage.get("text", ""))
        text = str(row.get("text", ""))
        start = _aligned_start(passage_text, declared_start, text)
        if start is None or not text:
            continue
        offset = int(passage["offset"]) + start
        if (offset, len(text)) in known_spans:
            continue
        if exact_text_only:
            identifier = ontology.resolve_alias(text, id_frequency=id_frequency)
        else:
            link = ontology.link(
                [str(row.get("canonical", "")), text],
                id_frequency=id_frequency,
                score_cutoff=fuzzy_cutoff,
            )
            identifier = link.identifier if link is not None else None
        if identifier is None:
            continue
        additions.append(
            {
                "identifier": identifier,
                "type": "Phenotype",
                "offset": offset,
                "length": len(text),
                "text": text,
                "note": "NO" if row.get("negated") is True else None,
            }
        )
    return merge_entities(additions)


def _fewshot_passage_examples(
    training_documents: list[JsonObject],
    target_section: str,
    *,
    max_examples: int = 5,
    max_chars: int = 1800,
) -> list[JsonObject]:
    examples: list[JsonObject] = []
    for document in training_documents:
        for passage_index, passage in enumerate(document.get("full_text", [])):
            if str(passage.get("section_type")) != target_section:
                continue
            text = str(passage.get("text", ""))
            if not text or len(text) > max_chars:
                continue
            start = int(passage["offset"])
            end = start + len(text)
            entities = []
            for entity in document.get("entities", []):
                offset = int(entity["offset"])
                if not start <= offset < end:
                    continue
                entities.append(
                    {
                        "start": offset - start,
                        "text": entity["text"],
                        "identifier": entity["identifier"],
                        "negated": entity.get("note") == "NO",
                    }
                )
            if not entities:
                continue
            examples.append(
                {
                    "section": target_section,
                    "text": text,
                    "entities": entities,
                }
            )
            if len(examples) >= max_examples:
                return examples
    return examples


def _fewshot_entity_prompt(
    document: JsonObject,
    chunks: list[tuple[int, JsonObject]],
    known_entities: list[JsonObject],
    examples: list[JsonObject],
) -> str:
    target = []
    for passage_index, passage in chunks:
        passage_start = int(passage["offset"])
        passage_text = str(passage.get("text", ""))
        target.append(
            {
                "passage_index": passage_index,
                "section": passage.get("section_type"),
                "text": passage_text,
                "already_found": [
                    {
                        "start": int(entity["offset"]) - passage_start,
                        "text": entity["text"],
                        "identifier": entity["identifier"],
                    }
                    for entity in known_entities
                    if passage_start <= int(entity["offset"]) < passage_start + len(passage_text)
                ],
            }
        )
    return f"""PMC_ID: {document['pmc_id']}

GOLD-STYLE EXAMPLES FROM OTHER ARTICLES:
{json.dumps(examples, ensure_ascii=False, separators=(',', ':'))}

TARGET PASSAGE CHUNK:
{json.dumps(target, ensure_ascii=False, separators=(',', ':'))}

Return exactly:
{{"entities":[{{"passage_index":0,"start":12,"text":"exact text","identifier":"HP:0000000","negated":false}}]}}
Return only additional entities not already_found. A target passage may have no additions.
"""


def _fewshot_identifier(
    row: JsonObject,
    text: str,
    ontology: HpoOntology,
    *,
    id_frequency: dict[str, int] | None,
) -> str | None:
    # The remote model is useful for finding a missing span, but its generated
    # HPO IDs are not reliable enough to establish the benchmark label. Resolve
    # the source text locally before considering any model-generated wording.
    identifier = ontology.resolve_alias(text, id_frequency=id_frequency)
    if identifier:
        return identifier
    canonical = str(row.get("canonical", "")).strip()
    link = ontology.link([text, canonical], id_frequency=id_frequency, score_cutoff=92.0)
    return link.identifier if link is not None else None


def discover_entities_fewshot_with_llm(
    document: JsonObject,
    known_entities: list[JsonObject],
    training_documents: list[JsonObject],
    ontology: HpoOntology,
    client: BigModelClient,
    *,
    id_frequency: dict[str, int] | None = None,
    max_chars: int = 6500,
    max_examples: int = 5,
) -> list[JsonObject]:
    """Recover annotation-style spans using passage chunks and held-out gold examples."""
    passages = list(enumerate(document.get("full_text", [])))
    chunks: list[list[tuple[int, JsonObject]]] = []
    current: list[tuple[int, JsonObject]] = []
    size = 0
    for item in passages:
        passage_size = len(str(item[1].get("text", ""))) + 180
        if current and size + passage_size > max_chars:
            chunks.append(current)
            current = []
            size = 0
        current.append(item)
        size += passage_size
    if current:
        chunks.append(current)

    known_spans = {
        (int(entity["offset"]), int(entity["length"])) for entity in known_entities
    }
    additions: list[JsonObject] = []
    for chunk in chunks:
        section = str(chunk[0][1].get("section_type", "UNKNOWN"))
        examples = _fewshot_passage_examples(
            training_documents, section, max_examples=max_examples
        )
        try:
            response = client.chat_json(
                [
                    {"role": "system", "content": _FEWSHOT_ENTITY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _fewshot_entity_prompt(
                            document, chunk, known_entities, examples
                        ),
                    },
                ],
                max_tokens=5000,
            )
        except (json.JSONDecodeError, RuntimeError):
            continue
        rows = response.get("entities", []) if isinstance(response, dict) else []
        passages_by_index = {index: passage for index, passage in chunk}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                passage_index = int(row["passage_index"])
                declared_start = int(row["start"])
            except (KeyError, TypeError, ValueError):
                continue
            passage = passages_by_index.get(passage_index)
            if passage is None:
                continue
            passage_text = str(passage.get("text", ""))
            text = str(row.get("text", ""))
            start = _aligned_start(passage_text, declared_start, text)
            if start is None or not text:
                continue
            offset = int(passage["offset"]) + start
            if (offset, len(text)) in known_spans:
                continue
            identifier = _fewshot_identifier(
                row, text, ontology, id_frequency=id_frequency
            )
            if identifier is None:
                continue
            additions.append(
                {
                    "identifier": identifier,
                    "type": "Phenotype",
                    "offset": offset,
                    "length": len(text),
                    "text": text,
                    "note": "NO" if row.get("negated") is True else None,
                }
            )
    return merge_entities(additions)
