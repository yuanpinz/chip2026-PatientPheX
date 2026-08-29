from __future__ import annotations

from patientphex_solver.association import (
    _filter_selected_indices_by_structure,
    filter_associations_by_structure,
)
from patientphex_solver.entities import GazetteerExtractor
from patientphex_solver.evaluation import evaluate
from patientphex_solver.io import validate_submission
from patientphex_solver.llm import parse_json_response
from patientphex_solver.ontology import HpoOntology


def test_parse_json_response_with_markdown_fence() -> None:
    assert parse_json_response("```json\n{\"ok\": true}\n```") == {"ok": True}


def test_hpo_alias_and_alt_id_resolution(tmp_path) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2\n\n[Term]\nid: HP:0000118\nname: Phenotypic abnormality\n\n[Term]\nid: HP:0000002\nname: Abnormality of body height\nsynonym: \"Short stature\" EXACT []\nis_a: HP:0000118 ! root\n""",
        encoding="utf-8",
    )
    ontology = HpoOntology.from_obo(obo)
    assert ontology.resolve_alias("short stature") == "HP:0000002"
    assert "HP:0000002" in ontology.descendants


def test_trained_abbreviation_requires_matching_expansion(tmp_path) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0000717
name: Autism
synonym: "Autism spectrum disorder" EXACT []
is_a: HP:0000118 ! root

[Term]
id: HP:0001249
name: Intellectual disability
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )
    ontology = HpoOntology.from_obo(obo)
    training = [
        {
            "pmc_id": "train",
            "full_text": [
                {
                    "offset": 0,
                    "text": "Autism spectrum disorder (ASD) was observed.",
                }
            ],
            "entities": [
                {
                    "identifier": "HP:0000717",
                    "offset": 27,
                    "length": 3,
                    "text": "ASD",
                    "note": None,
                }
            ],
        }
    ]
    extractor = GazetteerExtractor(ontology, training)
    expanded = {
        "pmc_id": "expanded",
        "full_text": [
            {
                "offset": 0,
                "text": "Autism spectrum disorder (ASD) was observed.",
            }
        ],
    }
    unrelated = {
        "pmc_id": "unrelated",
        "full_text": [{"offset": 0, "text": "The Gene ID was recorded."}],
    }
    assert "ASD" in [item["text"] for item in extractor.extract_document(expanded)]
    assert extractor.extract_document(unrelated) == []


def test_evaluation_counts_negative_prediction_as_false_positive() -> None:
    gold = [
        {
            "pmc_id": "1",
            "entities": [
                {
                    "identifier": "HP:0000002",
                    "offset": 0,
                    "length": 5,
                    "note": "NO",
                }
            ],
            "association": [],
        }
    ]
    predicted = [
        {
            "pmc_id": "1",
            "entities": [
                {
                    "identifier": "HP:0000002",
                    "offset": 0,
                    "length": 5,
                    "note": None,
                }
            ],
            "association": [],
        }
    ]
    assert evaluate(gold, predicted).mention.fp == 1


def test_submission_validation_accepts_empty_prediction_fields() -> None:
    expected = [
        {
            "pmc_id": "1",
            "pmid": None,
            "patient": [{"patient_id": "P1"}],
            "full_text": [],
        }
    ]
    predicted = [
        {
            "pmc_id": "1",
            "pmid": None,
            "entities": [],
            "association": [{"patient_id": "P1", "phenotype": []}],
        }
    ]
    assert validate_submission(predicted, expected) == []


def test_parse_json_response_recovers_truncated_assignments() -> None:
    assert parse_json_response('{"assignments":{"P1":[1,2],"P2":[3]') == {
        "assignments": {"P1": [1, 2], "P2": [3]}
    }


def test_structure_filter_keeps_patient_local_findings() -> None:
    document = {
        "pmc_id": "1",
        "patient": [
            {"patient_id": "P1", "mention": [{"offset": 0, "length": 2}]},
            {"patient_id": "P2", "mention": [{"offset": 17, "length": 2}]},
        ],
        "full_text": [
            {
                "section_type": "CASE",
                "offset": 0,
                "text": "P1 has seizures. P2 has ataxia.",
            }
        ],
    }
    entities = [
        {
            "identifier": "HP:0001250",
            "offset": 7,
            "length": 7,
            "text": "seizures",
            "note": None,
        },
        {
            "identifier": "HP:0001251",
            "offset": 24,
            "length": 6,
            "text": "ataxia",
            "note": None,
        },
    ]
    associations = [
        {"patient_id": "P1", "phenotype": ["HP:0001250", "HP:0001251"]},
        {"patient_id": "P2", "phenotype": ["HP:0001250", "HP:0001251"]},
    ]
    assert filter_associations_by_structure(document, entities, associations) == [
        {"patient_id": "P1", "phenotype": ["HP:0001250"]},
        {"patient_id": "P2", "phenotype": ["HP:0001251"]},
    ]


def test_structure_filter_is_occurrence_aware_for_repeated_hpo() -> None:
    document = {
        "pmc_id": "1",
        "patient": [
            {"patient_id": "P1", "mention": [{"offset": 0, "length": 2}]},
            {"patient_id": "P2", "mention": [{"offset": 17, "length": 2}]},
        ],
        "full_text": [
            {
                "section_type": "CASE",
                "offset": 0,
                "text": "P1 has seizures. P2 has seizures.",
            }
        ],
    }
    entities = [
        {
            "identifier": "HP:0001250",
            "offset": 7,
            "length": 8,
            "text": "seizures",
            "note": None,
        },
        {
            "identifier": "HP:0001250",
            "offset": 24,
            "length": 8,
            "text": "seizures",
            "note": None,
        },
    ]
    selected = {"P1": {0, 1}, "P2": {0, 1}}
    assert _filter_selected_indices_by_structure(document, entities, selected) == {
        "P1": {0},
        "P2": {1},
    }
