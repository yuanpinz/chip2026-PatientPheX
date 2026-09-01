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


def _entity_values(row: JsonObject) -> set[str]:
    values: set[str] = set()
    for entity in row.get("entities", []):
        if entity.get("note") == "NO":
            continue
        identifier = str(entity.get("identifier", ""))
        if identifier == "-1":
            values.add(str(entity.get("text", "")))
        else:
            values.add(identifier)
            values.update(identifier.split(";"))
    return values


def clip_associations_to_entities(rows: list[JsonObject]) -> list[JsonObject]:
    """Keep association values that are represented by positive entity labels."""
    predictions: list[JsonObject] = []
    for row in rows:
        allowed = _entity_values(row)
        associations = [
            {
                "patient_id": item["patient_id"],
                "phenotype": [
                    str(value)
                    for value in item.get("phenotype", [])
                    if str(value) in allowed
                ],
            }
            for item in row.get("association", [])
        ]
        predictions.append({**row, "association": associations})
    return predictions


def fuse_associations_by_patient_count(
    documents: list[JsonObject],
    base_rows: list[JsonObject],
    primary_rows: list[JsonObject],
    secondary_rows: list[JsonObject],
    *,
    union_multi: bool = False,
    union_patient_count_range: tuple[int, int] | None = None,
    max_primary_to_secondary_ratio: float | None = None,
    structure_previous_distance: int | None = None,
    structure_next_distance: int | None = None,
) -> list[JsonObject]:
    """Fuse two association predictions using a validated patient-count policy.

    Single-patient articles use the union of primary and secondary predictions.
    By default, multi-patient articles use only the secondary joint prediction
    so patients compete for values in one model call. ``union_multi`` enables a
    validated experimental policy that unions both sources for all articles.
    ``union_patient_count_range`` overrides both policies and unions sources only
    when the article patient count is within the inclusive range.
    When ``max_primary_to_secondary_ratio`` is set, a patient that would use the
    union instead uses only the non-empty secondary values when its primary
    value count divided by its secondary value count reaches the threshold.
    This protects the union from a primary model that produces substantially
    more candidates for one patient while leaving other patients unaffected.
    Entity annotations always come from the supplied base rows. Optional local
    structure filtering is applied after fusion.
    """
    if union_patient_count_range is not None:
        minimum, maximum = union_patient_count_range
        if minimum < 1 or maximum < minimum:
            raise ValueError(
                "union patient count range must satisfy 1 <= minimum <= maximum"
            )
    if (
        max_primary_to_secondary_ratio is not None
        and max_primary_to_secondary_ratio <= 0
    ):
        raise ValueError("primary-to-secondary ratio must be positive")

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
        patient_count = len(document.get("patient", []))
        multi_patient = patient_count > 1
        use_union = (
            union_patient_count_range[0]
            <= patient_count
            <= union_patient_count_range[1]
            if union_patient_count_range is not None
            else union_multi or not multi_patient
        )
        associations: list[JsonObject] = []
        for patient in document.get("patient", []):
            patient_id = str(patient["patient_id"])
            primary_values = primary_sets.get(patient_id, set())
            secondary_values = secondary_sets.get(patient_id, set())
            use_secondary_only = False
            if (
                use_union
                and max_primary_to_secondary_ratio is not None
                and secondary_values
            ):
                secondary_count = len(secondary_values)
                primary_count = len(primary_values)
                ratio = primary_count / secondary_count
                use_secondary_only = ratio >= max_primary_to_secondary_ratio
            selected = (
                secondary_values
                if not use_union or use_secondary_only
                else primary_values | secondary_values
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


def augment_associations_by_vote(
    documents: list[JsonObject],
    base_rows: list[JsonObject],
    source_rows: list[list[JsonObject]],
    *,
    min_votes: int = 1,
    structure_previous_distance: int | None = None,
    structure_next_distance: int | None = None,
    structure_wide_sections: set[str] | None = None,
    structure_wide_previous_distance: int | None = None,
    structure_wide_next_distance: int | None = None,
    propagate_explicit_groups: bool = False,
) -> list[JsonObject]:
    """Add voted associations from extra candidates to an existing prediction."""

    voted = fuse_associations_by_vote(
        documents,
        base_rows,
        source_rows,
        min_votes=min_votes,
        structure_previous_distance=structure_previous_distance,
        structure_next_distance=structure_next_distance,
        structure_wide_sections=structure_wide_sections,
        structure_wide_previous_distance=structure_wide_previous_distance,
        structure_wide_next_distance=structure_wide_next_distance,
        propagate_explicit_groups=propagate_explicit_groups,
    )
    base_by_id = _rows_by_id(base_rows)
    voted_by_id = _rows_by_id(voted)
    predictions: list[JsonObject] = []
    for document in documents:
        pmc_id = str(document["pmc_id"])
        try:
            base = base_by_id[pmc_id]
            extra = voted_by_id[pmc_id]
        except KeyError as exc:
            raise ValueError(f"missing augmentation input for PMC {pmc_id}") from exc
        extra_values = {
            str(item["patient_id"]): list(item.get("phenotype", []))
            for item in extra.get("association", [])
        }
        associations: list[JsonObject] = []
        for base_association in base.get("association", []):
            patient_id = str(base_association["patient_id"])
            ordered = list(base_association.get("phenotype", []))
            for value in extra_values.get(patient_id, []):
                if value not in ordered:
                    ordered.append(value)
            associations.append({"patient_id": patient_id, "phenotype": ordered})
        predictions.append(
            {
                "pmc_id": base["pmc_id"],
                "pmid": base.get("pmid"),
                "entities": list(base.get("entities", [])),
                "association": associations,
            }
        )
    return predictions
