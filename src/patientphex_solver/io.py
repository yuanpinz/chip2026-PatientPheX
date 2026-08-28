from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def read_jsonl(path: str | Path) -> list[JsonObject]:
    source = Path(path)
    rows: list[JsonObject] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"{source}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[JsonObject]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(destination)


def passage_for_offset(document: JsonObject, offset: int) -> tuple[int, JsonObject] | None:
    for index, passage in enumerate(document.get("full_text", [])):
        start = int(passage["offset"])
        end = start + len(passage.get("text", ""))
        if start <= offset < end:
            return index, passage
    return None


def validate_submission(
    predictions: list[JsonObject], expected: list[JsonObject]
) -> list[str]:
    errors: list[str] = []
    expected_by_id = {str(row.get("pmc_id")): row for row in expected}
    seen: set[str] = set()

    if len(predictions) != len(expected):
        errors.append(
            f"document count mismatch: predicted {len(predictions)}, expected {len(expected)}"
        )

    for row_index, row in enumerate(predictions, 1):
        pmc_id = str(row.get("pmc_id"))
        if pmc_id in seen:
            errors.append(f"row {row_index}: duplicate pmc_id {pmc_id}")
        seen.add(pmc_id)
        source = expected_by_id.get(pmc_id)
        if source is None:
            errors.append(f"row {row_index}: unexpected pmc_id {pmc_id}")
            continue
        if row.get("pmid") != source.get("pmid"):
            errors.append(f"{pmc_id}: pmid differs from input")

        entities = row.get("entities")
        if not isinstance(entities, list):
            errors.append(f"{pmc_id}: entities must be a list")
            continue
        entity_keys: set[tuple[int, int, str, str | None]] = set()
        for entity_index, entity in enumerate(entities):
            prefix = f"{pmc_id}: entity {entity_index}"
            if not isinstance(entity, dict):
                errors.append(f"{prefix} must be an object")
                continue
            required = {"identifier", "type", "offset", "length", "text", "note"}
            missing = required.difference(entity)
            if missing:
                errors.append(f"{prefix} missing fields: {sorted(missing)}")
                continue
            if entity["type"] != "Phenotype":
                errors.append(f"{prefix} has invalid type {entity['type']!r}")
            try:
                offset = int(entity["offset"])
                length = int(entity["length"])
            except (TypeError, ValueError):
                errors.append(f"{prefix} offset and length must be integers")
                continue
            located = passage_for_offset(source, offset)
            if located is None:
                errors.append(f"{prefix} offset {offset} is outside all passages")
            else:
                _, passage = located
                local_start = offset - int(passage["offset"])
                actual = passage["text"][local_start : local_start + length]
                if actual != entity["text"]:
                    errors.append(
                        f"{prefix} text mismatch: stored {entity['text']!r}, actual {actual!r}"
                    )
            key = (offset, length, str(entity["identifier"]), entity.get("note"))
            if key in entity_keys:
                errors.append(f"{prefix} duplicates an earlier entity")
            entity_keys.add(key)

        associations = row.get("association")
        if not isinstance(associations, list):
            errors.append(f"{pmc_id}: association must be a list")
            continue
        expected_patients = [str(item["patient_id"]) for item in source.get("patient", [])]
        actual_patients = [str(item.get("patient_id")) for item in associations]
        if actual_patients != expected_patients:
            errors.append(
                f"{pmc_id}: association patient order/IDs differ: "
                f"got {actual_patients}, expected {expected_patients}"
            )
        for association_index, association in enumerate(associations):
            phenotypes = association.get("phenotype") if isinstance(association, dict) else None
            if not isinstance(phenotypes, list) or not all(
                isinstance(value, str) for value in phenotypes
            ):
                errors.append(
                    f"{pmc_id}: association {association_index} phenotype must be a string list"
                )

    missing_ids = sorted(set(expected_by_id).difference(seen))
    if missing_ids:
        errors.append(f"missing pmc_id values: {missing_ids}")
    return errors
