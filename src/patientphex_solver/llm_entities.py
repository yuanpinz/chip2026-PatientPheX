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


def _entity_prompt(document: JsonObject, known_entities: list[JsonObject]) -> str:
    return f"""PMC_ID: {document['pmc_id']}

The caller sends one passage chunk at a time. The chunk uses a local zero-based start.

PASSAGE CHUNK AND ALREADY FOUND SPANS:
{{chunk}}

Return only additional phenotype spans not already_found, using this exact schema:
{{"entities":[{{"passage_index":0,"start":12,"text":"exact text","canonical":"canonical phenotype phrase","negated":false}}]}}
If nothing is missing, return {{"entities":[]}}.
"""


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
        response = client.chat_json(
            [
                {"role": "system", "content": _ENTITY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2200,
        )
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
