from __future__ import annotations

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
