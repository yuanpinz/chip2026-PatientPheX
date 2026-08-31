from __future__ import annotations

from typing import Any

from .association import filter_associations_by_structure

JsonObject = dict[str, Any]


def _rows_by_id(rows: list[JsonObject]) -> dict[str, JsonObject]:
    return {str(row["pmc_id"]): row for row in rows}


def _association_sets(row: JsonObject) -> dict[str, set[str]]:
    return {
        str(item["patient_id"]): set(item.get("phenotype", []))
        for item in row.get("association", [])
    }


def fuse_associations_by_patient_count(
    documents: list[JsonObject],
    base_rows: list[JsonObject],
    primary_rows: list[JsonObject],
    secondary_rows: list[JsonObject],
    *,
    union_multi: bool = False,
    structure_previous_distance: int | None = None,
    structure_next_distance: int | None = None,
) -> list[JsonObject]:
    """Fuse two association predictions using a validated patient-count policy.

    Single-patient articles use the union of primary and secondary predictions.
    By default, multi-patient articles use only the secondary joint prediction
    so patients compete for values in one model call. ``union_multi`` enables a
    validated experimental policy that unions both sources for all articles.
    Entity annotations always come from the supplied base rows. Optional local
    structure filtering is applied after fusion.
    """
    base_by_id = _rows_by_id(base_rows)
    primary_by_id = _rows_by_id(primary_rows)
    secondary_by_id = _rows_by_id(secondary_rows)
    predictions: list[JsonObject] = []
    for document in documents:
        pmc_id = str(document["pmc_id"])
        try:
            base = base_by_id[pmc_id]
            primary = primary_by_id[pmc_id]
            secondary = secondary_by_id[pmc_id]
        except KeyError as exc:
            raise ValueError(f"missing fusion input for PMC {pmc_id}") from exc

        primary_sets = _association_sets(primary)
        secondary_sets = _association_sets(secondary)
        multi_patient = len(document.get("patient", [])) > 1
        associations: list[JsonObject] = []
        for patient in document.get("patient", []):
            patient_id = str(patient["patient_id"])
            secondary_values = secondary_sets.get(patient_id, set())
            selected = (
                primary_sets.get(patient_id, set()) | secondary_values
                if union_multi
                else secondary_values
                if multi_patient
                else primary_sets.get(patient_id, set()) | secondary_values
            )
            ordered: list[str] = []
            for source in (primary, secondary):
                for item in source.get("association", []):
                    if str(item.get("patient_id")) != patient_id:
                        continue
                    for value in item.get("phenotype", []):
                        if value in selected and value not in ordered:
                            ordered.append(value)
            associations.append({"patient_id": patient_id, "phenotype": ordered})

        if (
            structure_previous_distance is not None
            or structure_next_distance is not None
        ):
            associations = filter_associations_by_structure(
                document,
                list(base.get("entities", [])),
                associations,
                previous_distance=structure_previous_distance or 1500,
                next_distance=structure_next_distance or 300,
            )

        predictions.append(
            {
                "pmc_id": base["pmc_id"],
                "pmid": base.get("pmid"),
                "entities": list(base.get("entities", [])),
                "association": associations,
            }
        )
    return predictions
