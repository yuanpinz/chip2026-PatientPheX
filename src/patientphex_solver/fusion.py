from __future__ import annotations

from typing import Any

from .association import (
    filter_associations_by_structure,
    propagate_explicit_group_associations,
)

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


def fuse_associations_by_vote(
    documents: list[JsonObject],
    base_rows: list[JsonObject],
    source_rows: list[list[JsonObject]],
    *,
    min_votes: int = 2,
    single_patient_source_indices: list[int] | None = None,
    single_patient_min_votes: int | None = None,
    structure_previous_distance: int | None = None,
    structure_next_distance: int | None = None,
    structure_wide_sections: set[str] | None = None,
    structure_wide_previous_distance: int | None = None,
    structure_wide_next_distance: int | None = None,
    propagate_explicit_groups: bool = False,
) -> list[JsonObject]:
    """Fuse several association predictions with a per-patient vote.

    A phenotype value is retained for a patient when at least ``min_votes``
    source predictions contain it. Source order provides deterministic output
    ordering. Optional structure filtering removes multi-patient assignments
    unsupported by local article anchors.
    """

    if not source_rows:
        raise ValueError("at least one association source is required")
    base_by_id = _rows_by_id(base_rows)
    source_by_id = [_rows_by_id(rows) for rows in source_rows]
    if min_votes < 1 or min_votes > len(source_rows):
        raise ValueError("min_votes must be between 1 and the number of sources")
    if single_patient_source_indices is not None:
        if not single_patient_source_indices or any(
            index < 0 or index >= len(source_rows)
            for index in single_patient_source_indices
        ):
            raise ValueError("single-patient source index is out of range")
        if len(set(single_patient_source_indices)) != len(
            single_patient_source_indices
        ):
            raise ValueError("single-patient source indices must be unique")
        single_votes = (
            single_patient_min_votes
            if single_patient_min_votes is not None
            else min_votes
        )
        if single_votes < 1 or single_votes > len(single_patient_source_indices):
            raise ValueError(
                "single-patient min votes must be between 1 and its source count"
            )

    predictions: list[JsonObject] = []
    for document in documents:
        pmc_id = str(document["pmc_id"])
        try:
            base = base_by_id[pmc_id]
            sources = [mapping[pmc_id] for mapping in source_by_id]
        except KeyError as exc:
            raise ValueError(f"missing vote input for PMC {pmc_id}") from exc

        source_sets = [_association_sets(row) for row in sources]
        selected_source_indices = (
            single_patient_source_indices
            if len(document.get("patient", [])) == 1
            and single_patient_source_indices is not None
            else list(range(len(sources)))
        )
        selected_sources = [sources[index] for index in selected_source_indices]
        selected_source_sets = [
            source_sets[index] for index in selected_source_indices
        ]
        is_single_subset = (
            len(document.get("patient", [])) == 1
            and single_patient_source_indices is not None
        )
        required_votes = (
            single_patient_min_votes
            if is_single_subset and single_patient_min_votes is not None
            else min_votes
        )
        associations: list[JsonObject] = []
        for patient in document.get("patient", []):
            patient_id = str(patient["patient_id"])
            values_by_source = [
                sets.get(patient_id, set()) for sets in selected_source_sets
            ]
            counts: dict[str, int] = {}
            ordered: list[str] = []
            for row in selected_sources:
                for item in row.get("association", []):
                    if str(item.get("patient_id")) != patient_id:
                        continue
                    for value in item.get("phenotype", []):
                        value = str(value)
                        if value not in counts:
                            ordered.append(value)
                        counts[value] = sum(value in values for values in values_by_source)
            associations.append(
                {
                    "patient_id": patient_id,
                    "phenotype": [
                        value
                        for value in ordered
                        if counts[value] >= required_votes
                    ],
                }
            )

        if (
            structure_previous_distance is not None
            or structure_next_distance is not None
        ):
            associations = filter_associations_by_structure(
                document,
                list(base.get("entities", [])),
                associations,
                previous_distance=(
                    structure_previous_distance
                    if structure_previous_distance is not None
                    else 1500
                ),
                next_distance=(
                    structure_next_distance
                    if structure_next_distance is not None
                    else 300
                ),
                wide_sections=structure_wide_sections,
                wide_previous_distance=structure_wide_previous_distance,
                wide_next_distance=structure_wide_next_distance,
            )

        if propagate_explicit_groups:
            associations = propagate_explicit_group_associations(
                document,
                list(base.get("entities", [])),
                associations,
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
