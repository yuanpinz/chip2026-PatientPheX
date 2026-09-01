from __future__ import annotations

import json

import pytest

from patientphex_solver.abbreviations import discover_abbreviation_entities
from patientphex_solver.association import (
    _filter_selected_indices_by_structure,
    filter_associations_by_structure,
    propagate_explicit_group_associations,
)
from patientphex_solver.association_judge import (
    associate_values_calibrated_with_llm,
    associate_values_joint_calibrated_with_llm,
    build_association_calibration_examples,
)
from patientphex_solver.cnn_fusion import (
    CnnFusionConfig,
    cnn_additions,
    fuse_cnn_entities,
)
from patientphex_solver.entities import (
    GazetteerExtractor,
    merge_entities,
    select_entities_by_vote,
    subtract_entities,
    vote_entities,
)
from patientphex_solver.entity_judge import (
    build_calibration_examples,
    judge_entities_with_llm,
)
from patientphex_solver.evaluation import evaluate
from patientphex_solver.fusion import (
    augment_associations_by_vote,
    clip_associations_to_entities,
    fuse_associations_by_patient_count,
    fuse_associations_by_vote,
)
from patientphex_solver.io import validate_submission
from patientphex_solver.llm import BigModelClient, parse_json_response
from patientphex_solver.llm_entities import (
    discover_entities_article_with_llm,
    discover_entities_fewshot_with_llm,
)
from patientphex_solver.ontology import HpoOntology
from patientphex_solver.patient_phenotypes import (
    _identifier_for_finding,
    discover_patient_phenotypes_with_llm,
)


def test_parse_json_response_with_markdown_fence() -> None:
    assert parse_json_response('```json\n{"ok": true}\n```') == {"ok": True}


def test_chat_json_retries_and_discards_invalid_cached_response(
    tmp_path, monkeypatch
) -> None:
    client = BigModelClient(cache_dir=tmp_path, retries=2)
    messages = [{"role": "user", "content": "return JSON"}]
    payload = client._payload(
        messages,
        max_tokens=100,
        temperature=0.0,
        enable_thinking=False,
    )
    cache_path = client._cache_path(payload)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text('{"text":"not JSON"}', encoding="utf-8")
    responses = iter(["not JSON", '{"ok":true}'])
    monkeypatch.setattr(client, "chat", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr("patientphex_solver.llm.time.sleep", lambda _: None)

    assert client.chat_json(messages, max_tokens=100) == {"ok": True}
    assert not cache_path.exists()


def test_chat_json_retries_when_chat_returns_no_text(monkeypatch, tmp_path) -> None:
    client = BigModelClient(cache_dir=tmp_path, retries=2)
    messages = [{"role": "user", "content": "return JSON"}]
    responses = iter([RuntimeError("empty response"), '{"ok":true}'])

    def fake_chat(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(client, "chat", fake_chat)
    monkeypatch.setattr("patientphex_solver.llm.time.sleep", lambda _: None)

    assert client.chat_json(messages, max_tokens=100) == {"ok": True}


def test_cnn_fusion_filters_overlap_and_caps_repeated_ids() -> None:
    base = [
        {
            "identifier": "HP:1",
            "type": "Phenotype",
            "offset": 10,
            "length": 8,
            "text": "existing",
            "note": None,
        }
    ]
    raw = [
        {
            "identifier": "HP:2",
            "type": "Phenotype",
            "offset": 10,
            "length": 4,
            "text": "exis",
            "note": None,
            "score": 1.0,
        },
        {
            "identifier": "HP:2",
            "type": "Phenotype",
            "offset": 30,
            "length": 6,
            "text": "second",
            "note": None,
            "score": 0.9998,
        },
        {
            "identifier": "HP:2",
            "type": "Phenotype",
            "offset": 50,
            "length": 6,
            "text": "thirdx",
            "note": None,
            "score": 0.9999,
        },
    ]
    additions = cnn_additions(
        base,
        raw,
        CnnFusionConfig(min_score=0.9997, min_text_length=6, max_per_identifier=1),
    )
    assert [(item["offset"], item["text"]) for item in additions] == [(50, "thirdx")]


def test_cnn_fusion_preserves_association_and_strips_score() -> None:
    base = [
        {
            "pmc_id": "p1",
            "pmid": "m1",
            "entities": [],
            "association": [{"patient_id": "P1", "phenotype": ["HP:1"]}],
        }
    ]
    cnn = [
        {
            "pmc_id": "p1",
            "pmid": "m1",
            "entities": [
                {
                    "identifier": "HP:2",
                    "type": "Phenotype",
                    "offset": 0,
                    "length": 6,
                    "text": "finding",
                    "note": None,
                    "score": 1.0,
                }
            ],
        }
    ]
    result = fuse_cnn_entities(base, cnn)
    assert result[0]["association"] == base[0]["association"]
    assert result[0]["entities"] == [
        {
            "identifier": "HP:2",
            "type": "Phenotype",
            "offset": 0,
            "length": 6,
            "text": "finding",
            "note": None,
        }
    ]


def test_merge_entities_combines_multiple_sources_and_deduplicates() -> None:
    first = {
        "identifier": "HP:1",
        "type": "Phenotype",
        "offset": 10,
        "length": 5,
        "text": "first",
        "note": None,
    }
    second = {
        "identifier": "HP:2",
        "type": "Phenotype",
        "offset": 20,
        "length": 6,
        "text": "second",
        "note": None,
    }
    assert merge_entities([first], [first, second], []) == [first, second]


def test_vote_entities_requires_matching_source_support() -> None:
    base = [{"pmc_id": "p1", "pmid": "m1", "entities": [], "association": []}]
    first = {
        "identifier": "HP:1",
        "type": "Phenotype",
        "offset": 10,
        "length": 5,
        "text": "first",
        "note": None,
    }
    second = {**first}
    third = {**first, "offset": 20, "length": 6, "text": "longer"}
    source_rows = [
        [{"pmc_id": "p1", "entities": [first]}],
        [{"pmc_id": "p1", "entities": [second, third]}],
        [{"pmc_id": "p1", "entities": [third]}],
    ]
    result = vote_entities(base, source_rows, min_votes=2)
    assert [entity["text"] for entity in result[0]["entities"]] == ["first", "longer"]

    short_only = vote_entities(
        base,
        source_rows,
        min_votes=2,
        max_text_length=5,
    )
    assert [entity["text"] for entity in short_only[0]["entities"]] == ["first"]


def test_vote_entities_rejects_invalid_text_length() -> None:
    base = [{"pmc_id": "p1", "entities": [], "association": []}]
    with pytest.raises(ValueError, match="max_text_length"):
        vote_entities(
            base,
            [[{"pmc_id": "p1", "entities": []}]],
            min_votes=1,
            max_text_length=0,
        )


def test_parse_json_response_ignores_trailing_explanation() -> None:
    assert parse_json_response(
        '{"accepted_indices":[1]}\nI selected the explicit finding.'
    ) == {"accepted_indices": [1]}


def test_calibrated_entity_judge_selects_returned_indices(tmp_path) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0001250
name: Seizure
is_a: HP:0000118 ! root

[Term]
id: HP:0004322
name: Short stature
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )
    ontology = HpoOntology.from_obo(obo)
    training = [
        {
            "pmc_id": "train",
            "full_text": [
                {"offset": 0, "section_type": "CASE", "text": "Seizure was observed."}
            ],
            "entities": [
                {
                    "identifier": "HP:0001250",
                    "type": "Phenotype",
                    "offset": 0,
                    "length": 7,
                    "text": "Seizure",
                    "note": None,
                }
            ],
        }
    ]
    extractor = GazetteerExtractor(ontology, training)
    calibration = build_calibration_examples(training, extractor, ontology)
    document = {
        "pmc_id": "target",
        "full_text": [
            {
                "offset": 0,
                "section_type": "CASE",
                "text": "Seizure and short stature were observed.",
            }
        ],
    }
    candidates = [
        {
            "identifier": "HP:0001250",
            "type": "Phenotype",
            "offset": 0,
            "length": 7,
            "text": "Seizure",
            "note": None,
        },
        {
            "identifier": "HP:0004322",
            "type": "Phenotype",
            "offset": 12,
            "length": 13,
            "text": "short stature",
            "note": None,
        },
    ]

    requested_max_tokens = []

    class FakeClient:
        def chat_json(self, messages, *, max_tokens=8000):
            assert "CALIBRATION EXAMPLES" in messages[1]["content"]
            requested_max_tokens.append(max_tokens)
            return {"accepted_indices": [1], "uncertain_indices": [0]}

    assert judge_entities_with_llm(
        document,
        candidates,
        ontology,
        FakeClient(),
        calibration,
        max_tokens=4000,
    ) == [candidates[1]]
    assert requested_max_tokens == [4000]


def test_calibrated_association_judge_groups_occurrences(tmp_path) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0001250
name: Seizure
is_a: HP:0000118 ! root

[Term]
id: HP:0004322
name: Short stature
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )
    ontology = HpoOntology.from_obo(obo)
    entities = [
        {
            "identifier": "HP:0001250",
            "type": "Phenotype",
            "offset": 11,
            "length": 7,
            "text": "seizure",
            "note": None,
        },
        {
            "identifier": "HP:0001250",
            "type": "Phenotype",
            "offset": 23,
            "length": 7,
            "text": "seizure",
            "note": None,
        },
        {
            "identifier": "HP:0004322",
            "type": "Phenotype",
            "offset": 35,
            "length": 13,
            "text": "short stature",
            "note": None,
        },
    ]
    document = {
        "pmc_id": "train",
        "patient": [
            {"patient_id": "P1", "mention": [{"text": "The child", "offset": 0, "length": 9}]}
        ],
        "full_text": [
            {
                "offset": 0,
                "section_type": "CASE",
                "text": "The child: seizure; seizure; short stature.",
            }
        ],
        "entities": entities,
        "association": [{"patient_id": "P1", "phenotype": ["HP:0001250"]}],
    }
    calibration = build_association_calibration_examples([document], ontology)
    assert {item.accepted for item in calibration} == {True, False}

    class FakeClient:
        def chat_json(self, messages, *, max_tokens=8000):
            candidate_section = messages[1]["content"].split(
                "CANDIDATE PHENOTYPE VALUES:", 1
            )[1]
            assert candidate_section.count('"value":"HP:0001250"') == 1
            return {"associated_indices": [0], "uncertain_indices": [1]}

    association = associate_values_calibrated_with_llm(
        document,
        entities,
        ontology,
        FakeClient(),
        calibration,
    )
    assert association == [{"patient_id": "P1", "phenotype": ["HP:0001250"]}]


def test_fewshot_entity_discovery_keeps_exact_span_and_id(tmp_path) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0001250
name: Seizure
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )
    ontology = HpoOntology.from_obo(obo)
    example = {
        "pmc_id": "example",
        "full_text": [
            {"offset": 0, "section_type": "CASE", "text": "Seizure occurred."}
        ],
        "entities": [
            {
                "identifier": "HP:0001250",
                "type": "Phenotype",
                "offset": 0,
                "length": 7,
                "text": "Seizure",
                "note": None,
            }
        ],
    }
    document = {
        "pmc_id": "target",
        "full_text": [
            {
                "offset": 100,
                "section_type": "CASE",
                "text": "The child developed seizures.",
            }
        ],
    }

    class FakeClient:
        def chat_json(self, messages, *, max_tokens=8000):
            assert "GOLD-STYLE EXAMPLES" in messages[1]["content"]
            return {
                "entities": [
                    {
                        "passage_index": 0,
                        "start": 20,
                        "text": "seizures",
                        "identifier": "HP:0001250",
                        "negated": False,
                    }
                ]
            }

    additions = discover_entities_fewshot_with_llm(
        document, [], [example], ontology, FakeClient()
    )
    assert additions == [
        {
            "identifier": "HP:0001250",
            "type": "Phenotype",
            "offset": 120,
            "length": 8,
            "text": "seizures",
            "note": None,
        }
    ]


def test_joint_calibrated_association_returns_all_patient_keys(tmp_path) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0001250
name: Seizure
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )
    ontology = HpoOntology.from_obo(obo)
    document = {
        "pmc_id": "target",
        "patient": [
            {"patient_id": "P1", "mention": [{"text": "Patient 1", "offset": 0, "length": 9}]},
            {"patient_id": "P2", "mention": [{"text": "Patient 2", "offset": 20, "length": 9}]},
        ],
        "full_text": [
            {
                "offset": 0,
                "section_type": "CASE",
                "text": "Patient 1 had seizure. Patient 2 had no seizure.",
            }
        ],
    }
    entity = {
        "identifier": "HP:0001250",
        "type": "Phenotype",
        "offset": 15,
        "length": 7,
        "text": "seizure",
        "note": None,
    }

    class FakeClient:
        def chat_json(self, messages, *, max_tokens=8000):
            assert "LISTED PATIENTS" in messages[1]["content"]
            return {"assignments": {"P1": [0], "P2": []}}

    result = associate_values_joint_calibrated_with_llm(
        document,
        [entity],
        ontology,
        FakeClient(),
        [],
        structure_multi_patient=False,
    )
    assert result == [
        {"patient_id": "P1", "phenotype": ["HP:0001250"]},
        {"patient_id": "P2", "phenotype": []},
    ]


def test_joint_calibrated_association_skips_malformed_batch(tmp_path) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0001250
name: Seizure
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )
    ontology = HpoOntology.from_obo(obo)
    document = {
        "pmc_id": "malformed",
        "patient": [{"patient_id": "P1"}],
    }
    entity = {
        "identifier": "HP:0001250",
        "type": "Phenotype",
        "offset": 0,
        "length": 7,
        "text": "seizure",
        "note": None,
    }

    class FakeClient:
        def chat_json(self, messages, *, max_tokens=8000):
            raise json.JSONDecodeError("bad response", "not json", 0)

    assert associate_values_joint_calibrated_with_llm(
        document,
        [entity],
        ontology,
        FakeClient(),
        [],
        structure_multi_patient=False,
    ) == [{"patient_id": "P1", "phenotype": []}]


def test_fusion_unions_single_patient_and_uses_joint_multi_patient() -> None:
    documents = [
        {"pmc_id": "single", "patient": [{"patient_id": "P1"}]},
        {
            "pmc_id": "multi",
            "patient": [{"patient_id": "P1"}, {"patient_id": "P2"}],
        },
    ]
    base = [
        {"pmc_id": "single", "pmid": "1", "entities": [{"text": "a"}]},
        {"pmc_id": "multi", "pmid": "2", "entities": [{"text": "b"}]},
    ]
    primary = [
        {
            "pmc_id": "single",
            "association": [{"patient_id": "P1", "phenotype": ["A"]}],
        },
        {
            "pmc_id": "multi",
            "association": [
                {"patient_id": "P1", "phenotype": ["A"]},
                {"patient_id": "P2", "phenotype": ["B"]},
            ],
        },
    ]
    secondary = [
        {
            "pmc_id": "single",
            "association": [{"patient_id": "P1", "phenotype": ["B"]}],
        },
        {
            "pmc_id": "multi",
            "association": [
                {"patient_id": "P1", "phenotype": ["C"]},
                {"patient_id": "P2", "phenotype": []},
            ],
        },
    ]

    fused = fuse_associations_by_patient_count(documents, base, primary, secondary)
    assert fused[0]["association"] == [
        {"patient_id": "P1", "phenotype": ["A", "B"]}
    ]
    assert fused[1]["association"] == [
        {"patient_id": "P1", "phenotype": ["C"]},
        {"patient_id": "P2", "phenotype": []},
    ]


def test_select_entities_by_vote_can_replace_base_with_exact_intersection() -> None:
    base = [
        {
            "pmc_id": "doc",
            "pmid": "1",
            "entities": [{"offset": 1, "length": 1, "identifier": "HP:BASE"}],
            "association": [{"patient_id": "P1", "phenotype": ["HP:A", "HP:BASE"]}],
        }
    ]
    source_a = [
        {"pmc_id": "doc", "entities": [
            {"offset": 10, "length": 3, "identifier": "HP:A", "text": "one", "note": None},
            {"offset": 20, "length": 3, "identifier": "HP:B", "text": "two", "note": None},
            {"offset": 20, "length": 3, "identifier": "HP:B", "text": "two", "note": None},
        ]}
    ]
    source_b = [
        {"pmc_id": "doc", "entities": [
            {"offset": 10, "length": 3, "identifier": "HP:A", "text": "one", "note": None},
            {"offset": 30, "length": 3, "identifier": "HP:C", "text": "tri", "note": None},
        ]}
    ]

    selected = select_entities_by_vote(base, [source_a, source_b], min_votes=2)
    assert [(item["offset"], item["identifier"]) for item in selected[0]["entities"]] == [(10, "HP:A")]
    assert selected[0]["association"] == base[0]["association"]


def test_subtract_entities_keeps_only_unseen_candidate_occurrences() -> None:
    base = [
        {
            "pmc_id": "doc",
            "pmid": "1",
            "entities": [
                {
                    "offset": 10,
                    "length": 3,
                    "identifier": "HP:A",
                    "text": "one",
                    "note": None,
                }
            ],
            "association": [{"patient_id": "P1", "phenotype": ["HP:A"]}],
        }
    ]
    candidates = [
        {
            "pmc_id": "doc",
            "entities": [
                {
                    "offset": 10,
                    "length": 3,
                    "identifier": "HP:A",
                    "text": "one",
                    "note": None,
                },
                {
                    "offset": 20,
                    "length": 3,
                    "identifier": "HP:B",
                    "text": "two",
                    "note": None,
                },
            ],
        }
    ]

    result = subtract_entities(base, candidates)
    assert result == [
        {
            "pmc_id": "doc",
            "pmid": "1",
            "entities": [
                {
                    "offset": 20,
                    "length": 3,
                    "identifier": "HP:B",
                    "text": "two",
                    "note": None,
                }
            ],
            "association": [],
        }
    ]


def test_clip_associations_to_entities_supports_compound_ids_and_unmapped_text() -> None:
    rows = [
        {
            "pmc_id": "doc",
            "entities": [
                {"identifier": "HP:A;HP:B", "text": "finding", "note": None},
                {"identifier": "-1", "text": "unmapped finding", "note": None},
                {"identifier": "HP:NO", "text": "negative", "note": "NO"},
            ],
            "association": [
                {"patient_id": "P1", "phenotype": ["HP:A", "HP:B", "HP:NO", "unmapped finding", "missing"]}
            ],
        }
    ]
    clipped = clip_associations_to_entities(rows)
    assert clipped[0]["association"] == [
        {"patient_id": "P1", "phenotype": ["HP:A", "HP:B", "unmapped finding"]}
    ]


def test_fusion_can_union_multi_patient_sources() -> None:
    document = {
        "pmc_id": "multi",
        "patient": [{"patient_id": "P1"}, {"patient_id": "P2"}],
    }
    base = [{"pmc_id": "multi", "pmid": "1", "entities": []}]
    primary = [{"pmc_id": "multi", "association": [{"patient_id": "P1", "phenotype": ["A"]}, {"patient_id": "P2", "phenotype": []}]}]
    secondary = [{"pmc_id": "multi", "association": [{"patient_id": "P1", "phenotype": ["B"]}, {"patient_id": "P2", "phenotype": ["C"]}]}]
    fused = fuse_associations_by_patient_count(
        [document], base, primary, secondary, union_multi=True
    )
    assert fused[0]["association"] == [
        {"patient_id": "P1", "phenotype": ["A", "B"]},
        {"patient_id": "P2", "phenotype": ["C"]},
    ]


def test_fusion_can_union_only_for_patient_count_range() -> None:
    documents = [
        {"pmc_id": "single", "patient": [{"patient_id": "P1"}]},
        {
            "pmc_id": "multi",
            "patient": [{"patient_id": "P1"}, {"patient_id": "P2"}],
        },
    ]
    base = [
        {"pmc_id": "single", "pmid": "1", "entities": []},
        {"pmc_id": "multi", "pmid": "2", "entities": []},
    ]
    primary = [
        {
            "pmc_id": "single",
            "association": [{"patient_id": "P1", "phenotype": ["A"]}],
        },
        {
            "pmc_id": "multi",
            "association": [
                {"patient_id": "P1", "phenotype": ["A"]},
                {"patient_id": "P2", "phenotype": []},
            ],
        },
    ]
    secondary = [
        {
            "pmc_id": "single",
            "association": [{"patient_id": "P1", "phenotype": ["B"]}],
        },
        {
            "pmc_id": "multi",
            "association": [
                {"patient_id": "P1", "phenotype": ["B"]},
                {"patient_id": "P2", "phenotype": ["C"]},
            ],
        },
    ]

    fused = fuse_associations_by_patient_count(
        documents,
        base,
        primary,
        secondary,
        union_patient_count_range=(2, 7),
    )
    assert fused[0]["association"] == [
        {"patient_id": "P1", "phenotype": ["B"]}
    ]
    assert fused[1]["association"] == [
        {"patient_id": "P1", "phenotype": ["A", "B"]},
        {"patient_id": "P2", "phenotype": ["C"]},
    ]


def test_fusion_rejects_invalid_patient_count_range() -> None:
    with pytest.raises(ValueError, match="patient count range"):
        fuse_associations_by_patient_count(
            [],
            [],
            [],
            [],
            union_patient_count_range=(3, 2),
        )


def test_fusion_can_suppress_primary_outlier_per_patient() -> None:
    document = {
        "pmc_id": "multi",
        "patient": [
            {"patient_id": "P1"},
            {"patient_id": "P2"},
            {"patient_id": "P3"},
        ],
    }
    base = [{"pmc_id": "multi", "pmid": "1", "entities": []}]
    primary = [
        {
            "pmc_id": "multi",
            "association": [
                {"patient_id": "P1", "phenotype": ["A", "B", "C", "D"]},
                {"patient_id": "P2", "phenotype": ["E"]},
                {"patient_id": "P3", "phenotype": ["H"]},
            ],
        }
    ]
    secondary = [
        {
            "pmc_id": "multi",
            "association": [
                {"patient_id": "P1", "phenotype": ["A", "B"]},
                {"patient_id": "P2", "phenotype": ["F", "G"]},
                {"patient_id": "P3", "phenotype": []},
            ],
        }
    ]

    fused = fuse_associations_by_patient_count(
        [document],
        base,
        primary,
        secondary,
        union_patient_count_range=(2, 7),
        max_primary_to_secondary_ratio=2,
    )
    assert fused[0]["association"] == [
        {"patient_id": "P1", "phenotype": ["A", "B"]},
        {"patient_id": "P2", "phenotype": ["E", "F", "G"]},
        {"patient_id": "P3", "phenotype": ["H"]},
    ]


def test_fusion_rejects_non_positive_primary_secondary_ratio() -> None:
    with pytest.raises(ValueError, match="ratio must be positive"):
        fuse_associations_by_patient_count(
            [],
            [],
            [],
            [],
            max_primary_to_secondary_ratio=0,
        )


def test_fusion_votes_across_sources() -> None:
    document = {
        "pmc_id": "multi",
        "patient": [{"patient_id": "P1"}],
    }
    base = [{"pmc_id": "multi", "pmid": "1", "entities": []}]
    sources = [
        [{"pmc_id": "multi", "association": [{"patient_id": "P1", "phenotype": ["A", "B"]}]}],
        [{"pmc_id": "multi", "association": [{"patient_id": "P1", "phenotype": ["A", "C"]}]}],
        [{"pmc_id": "multi", "association": [{"patient_id": "P1", "phenotype": ["A"]}]}],
    ]
    fused = fuse_associations_by_vote(
        [document], base, sources, min_votes=2
    )
    assert fused[0]["association"] == [
        {"patient_id": "P1", "phenotype": ["A"]}
    ]


def test_fusion_uses_single_patient_source_subset_only_for_singletons() -> None:
    documents = [
        {"pmc_id": "single", "patient": [{"patient_id": "P1"}]},
        {
            "pmc_id": "multi",
            "patient": [{"patient_id": "P1"}, {"patient_id": "P2"}],
        },
    ]
    base = [
        {"pmc_id": "single", "pmid": "1", "entities": []},
        {"pmc_id": "multi", "pmid": "2", "entities": []},
    ]
    sources = [
        [
            {"pmc_id": "single", "association": [{"patient_id": "P1", "phenotype": ["A"]}]},
            {"pmc_id": "multi", "association": [{"patient_id": "P1", "phenotype": ["M"]}, {"patient_id": "P2", "phenotype": []}]},
        ],
        [
            {"pmc_id": "single", "association": [{"patient_id": "P1", "phenotype": ["A"]}]},
            {"pmc_id": "multi", "association": [{"patient_id": "P1", "phenotype": []}, {"patient_id": "P2", "phenotype": ["N"]}]},
        ],
        [
            {"pmc_id": "single", "association": [{"patient_id": "P1", "phenotype": ["B"]}]},
            {"pmc_id": "multi", "association": [{"patient_id": "P1", "phenotype": ["M"]}, {"patient_id": "P2", "phenotype": []}]},
        ],
        [
            {"pmc_id": "single", "association": [{"patient_id": "P1", "phenotype": ["C"]}]},
            {"pmc_id": "multi", "association": [{"patient_id": "P1", "phenotype": []}, {"patient_id": "P2", "phenotype": ["N"]}]},
        ],
    ]

    fused = fuse_associations_by_vote(
        documents,
        base,
        sources,
        min_votes=2,
        single_patient_source_indices=[0, 1, 3],
        single_patient_min_votes=2,
    )

    assert fused[0]["association"] == [
        {"patient_id": "P1", "phenotype": ["A"]}
    ]
    assert fused[1]["association"] == [
        {"patient_id": "P1", "phenotype": ["M"]},
        {"patient_id": "P2", "phenotype": ["N"]},
    ]


def test_fusion_rejects_invalid_single_patient_source_indices() -> None:
    document = {"pmc_id": "single", "patient": [{"patient_id": "P1"}]}
    base = [{"pmc_id": "single", "pmid": "1", "entities": []}]
    source = [{"pmc_id": "single", "association": [{"patient_id": "P1", "phenotype": []}]}]

    with pytest.raises(ValueError, match="out of range"):
        fuse_associations_by_vote(
            [document],
            base,
            [source],
            min_votes=1,
            single_patient_source_indices=[1],
        )
    with pytest.raises(ValueError, match="unique"):
        fuse_associations_by_vote(
            [document], base, [source, source], single_patient_source_indices=[0, 0]
        )
    with pytest.raises(ValueError, match="single-patient min votes"):
        fuse_associations_by_vote(
            [document],
            base,
            [source],
            min_votes=1,
            single_patient_source_indices=[0],
            single_patient_min_votes=0,
        )


def test_augment_associations_preserves_base_and_adds_voted_values() -> None:
    documents = [
        {"pmc_id": "single", "patient": [{"patient_id": "P1"}], "full_text": []}
    ]
    base = [
        {
            "pmc_id": "single",
            "pmid": None,
            "entities": [],
            "association": [{"patient_id": "P1", "phenotype": ["A"]}],
        }
    ]
    source = [
        {
            "pmc_id": "single",
            "association": [{"patient_id": "P1", "phenotype": ["B"]}],
        }
    ]
    augmented = augment_associations_by_vote(documents, base, [source])
    assert augmented[0]["association"] == [
        {"patient_id": "P1", "phenotype": ["A", "B"]}
    ]


def test_explicit_group_propagation_handles_both_listed_patients() -> None:
    text = "Both P1 and P2 had seizures."
    document = {
        "pmc_id": "group",
        "patient": [
            {"patient_id": "P1", "mention": [{"text": "P1", "offset": 5, "length": 2}]},
            {"patient_id": "P2", "mention": [{"text": "P2", "offset": 12, "length": 2}]},
        ],
        "full_text": [{"section_type": "CASE", "offset": 0, "text": text}],
    }
    entity = {
        "identifier": "HP:0001250",
        "offset": text.index("seizures"),
        "length": len("seizures"),
        "text": "seizures",
        "note": None,
    }
    associations = [
        {"patient_id": "P1", "phenotype": ["HP:0001250"]},
        {"patient_id": "P2", "phenotype": []},
    ]

    assert propagate_explicit_group_associations(
        document, [entity], associations
    ) == [
        {"patient_id": "P1", "phenotype": ["HP:0001250"]},
        {"patient_id": "P2", "phenotype": ["HP:0001250"]},
    ]


def test_explicit_group_propagation_does_not_copy_patient_specific_tail() -> None:
    text = "Both P1 and P2 had seizures, and P1 additionally has epilepsy."
    document = {
        "pmc_id": "tail",
        "patient": [
            {"patient_id": "P1", "mention": [{"text": "P1", "offset": 5, "length": 2}]},
            {"patient_id": "P2", "mention": [{"text": "P2", "offset": 12, "length": 2}]},
        ],
        "full_text": [{"section_type": "CASE", "offset": 0, "text": text}],
    }
    entity = {
        "identifier": "HP:0001250",
        "offset": text.index("epilepsy"),
        "length": len("epilepsy"),
        "text": "epilepsy",
        "note": None,
    }
    associations = [
        {"patient_id": "P1", "phenotype": []},
        {"patient_id": "P2", "phenotype": []},
    ]

    assert propagate_explicit_group_associations(
        document, [entity], associations
    ) == [
        {"patient_id": "P1", "phenotype": ["HP:0001250"]},
        {"patient_id": "P2", "phenotype": []},
    ]


def test_explicit_group_propagation_ignores_negated_shared_finding() -> None:
    text = "Both P1 and P2 had no seizures."
    document = {
        "pmc_id": "negated-group",
        "patient": [
            {"patient_id": "P1", "mention": [{"text": "P1", "offset": 5, "length": 2}]},
            {"patient_id": "P2", "mention": [{"text": "P2", "offset": 12, "length": 2}]},
        ],
        "full_text": [{"section_type": "CASE", "offset": 0, "text": text}],
    }
    entity = {
        "identifier": "HP:0001250",
        "offset": text.index("seizures"),
        "length": len("seizures"),
        "text": "seizures",
        "note": None,
    }
    associations = [
        {"patient_id": "P1", "phenotype": ["HP:0001250"]},
        {"patient_id": "P2", "phenotype": []},
    ]

    assert propagate_explicit_group_associations(
        document, [entity], associations
    ) == associations


def test_explicit_group_propagation_uses_previous_patient_enumeration() -> None:
    text = "P1 and P2 were examined. All had seizures."
    document = {
        "pmc_id": "enumerated-group",
        "patient": [
            {"patient_id": "P1", "mention": [{"text": "P1", "offset": 0, "length": 2}]},
            {"patient_id": "P2", "mention": [{"text": "P2", "offset": 7, "length": 2}]},
        ],
        "full_text": [{"section_type": "CASE", "offset": 0, "text": text}],
    }
    entity = {
        "identifier": "HP:0001250",
        "offset": text.index("seizures"),
        "length": len("seizures"),
        "text": "seizures",
        "note": None,
    }
    associations = [
        {"patient_id": "P1", "phenotype": []},
        {"patient_id": "P2", "phenotype": []},
    ]

    assert propagate_explicit_group_associations(
        document, [entity], associations
    ) == [
        {"patient_id": "P1", "phenotype": ["HP:0001250"]},
        {"patient_id": "P2", "phenotype": ["HP:0001250"]},
    ]


def test_structure_filter_widens_only_selected_sections() -> None:
    document = {
        "pmc_id": "wide",
        "patient": [
            {"patient_id": "P1", "mention": [{"offset": 0, "length": 2}]},
            {"patient_id": "P2", "mention": [{"offset": 3000, "length": 2}]},
        ],
        "full_text": [
            {"section_type": "CASE", "offset": 0, "text": "P1"},
            {"section_type": "CASE", "offset": 1000, "text": "Seizures were observed."},
            {"section_type": "CASE", "offset": 3000, "text": "P2"},
        ],
    }
    entity = {
        "identifier": "HP:0001250",
        "offset": 1000,
        "length": 8,
        "text": "Seizures",
        "note": None,
    }
    associations = [
        {"patient_id": "P1", "phenotype": ["HP:0001250"]},
        {"patient_id": "P2", "phenotype": []},
    ]

    assert filter_associations_by_structure(
        document, [entity], associations, previous_distance=800, next_distance=200
    ) == [
        {"patient_id": "P1", "phenotype": []},
        {"patient_id": "P2", "phenotype": []},
    ]
    assert filter_associations_by_structure(
        document,
        [entity],
        associations,
        previous_distance=800,
        next_distance=200,
        wide_sections={"RESULTS"},
        wide_previous_distance=5000,
        wide_next_distance=200,
    ) == [
        {"patient_id": "P1", "phenotype": []},
        {"patient_id": "P2", "phenotype": []},
    ]
    assert filter_associations_by_structure(
        document,
        [entity],
        associations,
        previous_distance=800,
        next_distance=200,
        wide_sections={"CASE"},
        wide_previous_distance=5000,
        wide_next_distance=200,
    ) == associations


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


def test_trained_abbreviation_requires_expansion_by_default(tmp_path) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0001149
name: Lattice corneal dystrophy
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )
    ontology = HpoOntology.from_obo(obo)
    training = [
        {
            "pmc_id": "train",
            "full_text": [
                {"offset": 0, "text": "LCD was repeatedly observed."}
            ],
            "entities": [
                {
                    "identifier": "HP:0001149",
                    "offset": 0,
                    "length": 3,
                    "text": "LCD",
                    "note": None,
                }
            ],
        }
    ]
    target = {"pmc_id": "target", "full_text": [{"offset": 0, "text": "LCD."}]}
    strict = GazetteerExtractor(ontology, training)
    assert strict.extract_document(target) == []


def test_exact_ontology_alias_overrides_conflicting_training_surface(tmp_path) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0000002
name: Abnormality of stature
synonym: "Small stature" EXACT []
is_a: HP:0000118 ! root

[Term]
id: HP:0004322
name: Short stature
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )
    ontology = HpoOntology.from_obo(obo)
    training = [
        {
            "pmc_id": "train",
            "full_text": [{"offset": 0, "text": "Small stature."}],
            "entities": [
                {
                    "identifier": "HP:0004322",
                    "offset": 0,
                    "length": 12,
                    "text": "Small stature",
                    "note": None,
                }
            ],
        }
    ]
    extractor = GazetteerExtractor(ontology, training)
    predicted = extractor.extract_document(
        {"pmc_id": "article", "full_text": [{"offset": 0, "text": "Small stature."}]}
    )
    assert predicted[0]["identifier"] == "HP:0000002"


def test_low_frequency_preferred_hpo_name_is_recovered(tmp_path) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0001250
name: Seizure
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )
    ontology = HpoOntology.from_obo(obo)
    training = [
        {
            "pmc_id": "train",
            "full_text": [
                {"offset": index * 20, "text": "Seizure."} for index in range(10)
            ],
            "entities": [
                {
                    "identifier": "HP:0001250",
                    "offset": 0,
                    "length": 7,
                    "text": "Seizure",
                    "note": None,
                }
            ],
        }
    ]
    extractor = GazetteerExtractor(ontology, training)
    assert extractor.surface_stats["seizure"].precision == 0.1
    assert extractor.alias_source["seizure"] == "recovered-train-hpo"
    predicted = extractor.extract_document(
        {"pmc_id": "article", "full_text": [{"offset": 0, "text": "Seizure."}]}
    )
    assert predicted[0]["identifier"] == "HP:0001250"


def test_negation_requires_a_direct_scope(tmp_path) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0001250
name: Seizure
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )
    ontology = HpoOntology.from_obo(obo)
    extractor = GazetteerExtractor(
        ontology,
        [
            {
                "pmc_id": "train",
                "full_text": [
                    {
                        "offset": 0,
                        "text": "Seizure.",
                    }
                ],
                "entities": [
                    {
                        "identifier": "HP:0001250",
                        "offset": 0,
                        "length": 7,
                        "text": "Seizure",
                        "note": None,
                    }
                ],
            }
        ],
    )
    predicted = extractor.extract_document(
        {
            "pmc_id": "article",
            "full_text": [
                {
                    "offset": 0,
                    "text": "No evidence of seizure. Seizure was absent.",
                }
            ],
        }
    )
    assert [item["note"] for item in predicted] == ["NO", "NO"]


def test_article_entity_discovery_aligns_offsets_and_requires_exact_text_alias(
    tmp_path,
) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0000739
name: Anxiety
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )

    class FakeClient:
        def chat_json(self, messages, *, max_tokens):
            return {
                "entities": [
                    {
                        "passage_index": 0,
                        "start": 0,
                        "text": "anxiety",
                        "canonical": "Anxiety",
                        "negated": False,
                    },
                    {
                        "passage_index": 0,
                        "start": 20,
                        "text": "not a phenotype",
                        "canonical": "Anxiety",
                        "negated": False,
                    },
                ]
            }

    ontology = HpoOntology.from_obo(obo)
    document = {
        "pmc_id": "article",
        "full_text": [
            {
                "section_type": "CASE",
                "offset": 0,
                "text": "Severe anxiety was observed.",
            }
        ],
    }
    additions = discover_entities_article_with_llm(
        document,
        [],
        ontology,
        FakeClient(),
    )
    assert additions == [
        {
            "identifier": "HP:0000739",
            "type": "Phenotype",
            "offset": 7,
            "length": 7,
            "text": "anxiety",
            "note": None,
        }
    ]


def test_direct_patient_phenotypes_require_exact_evidence_and_known_hpo(
    tmp_path,
) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0001250
name: Seizure
synonym: "Seizures" EXACT []
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )

    class FakeClient:
        def chat_json(self, messages, *, max_tokens):
            return {
                "assignments": {
                    "P1": [
                        {
                            "passage_index": 0,
                            "start": 0,
                            "text": "seizures",
                            "canonical": "Seizure",
                            "hpo_id": "HP:0001250",
                        },
                        {
                            "passage_index": 0,
                            "start": 40,
                            "text": "invented finding",
                            "canonical": "Seizure",
                            "hpo_id": "HP:0001250",
                        },
                    ]
                }
            }

    ontology = HpoOntology.from_obo(obo)
    extractor = GazetteerExtractor(ontology, [])
    document = {
        "pmc_id": "article",
        "patient": [{"patient_id": "P1", "mention": []}],
        "full_text": [
            {"section_type": "CASE", "offset": 10, "text": "The patient had seizures."}
        ],
    }
    additions, associations = discover_patient_phenotypes_with_llm(
        document, ontology, extractor, FakeClient()
    )
    assert additions == [
        {
            "identifier": "HP:0001250",
            "type": "Phenotype",
            "offset": 26,
            "length": 8,
            "text": "seizures",
            "note": None,
        }
    ]
    assert associations == [{"patient_id": "P1", "phenotype": ["HP:0001250"]}]


def test_direct_patient_finding_prefers_compatible_explicit_hpo_id(tmp_path) -> None:
    obo = tmp_path / "small.obo"
    obo.write_text(
        """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0001900
name: Hemoglobin abnormality
is_a: HP:0000118 ! root

[Term]
id: HP:0040217
name: Increased hemoglobin A1c
is_a: HP:0000118 ! root
""",
        encoding="utf-8",
    )
    ontology = HpoOntology.from_obo(obo)
    training = [
        {
            "pmc_id": "train",
            "full_text": [
                {"offset": 0, "text": "Increased hemoglobin A1c was observed."}
            ],
            "entities": [
                {
                    "identifier": "HP:0001900",
                    "offset": 0,
                    "length": 25,
                    "text": "Increased hemoglobin A1c",
                    "note": None,
                }
            ],
        }
    ]
    extractor = GazetteerExtractor(ontology, training)

    assert (
        _identifier_for_finding(
            {
                "text": "Increased hemoglobin A1c",
                "canonical": "Increased hemoglobin A1c",
                "hpo_id": "HP:0040217",
            },
            ontology,
            extractor,
        )
        == "HP:0040217"
    )


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


def test_parse_json_response_recovers_truncated_patient_findings() -> None:
    response = parse_json_response(
        '{"assignments":{"P1":['
        '{"passage_index":2,"start":4,"text":"seizures",'
        '"canonical":"Seizure","hpo_id":"HP:0001250"},'
        '{"passage_index":2,"start":20,"text":"unfinished"'
    )
    assert response == {
        "assignments": {
            "P1": [
                {
                    "passage_index": 2,
                    "start": 4,
                    "text": "seizures",
                    "canonical": "Seizure",
                    "hpo_id": "HP:0001250",
                }
            ]
        }
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


def test_structure_filter_sorts_patient_anchors_by_article_offset() -> None:
    document = {
        "pmc_id": "1",
        # Patient metadata order is independent of mention order in the article.
        "patient": [
            {"patient_id": "P2", "mention": [{"offset": 100, "length": 2}]},
            {"patient_id": "P1", "mention": [{"offset": 0, "length": 2}]},
        ],
        "full_text": [
            {"section_type": "CASE", "offset": 0, "text": "P1"},
            {"section_type": "CASE", "offset": 50, "text": "Ataxia was observed."},
            {"section_type": "CASE", "offset": 100, "text": "P2"},
        ],
    }
    entities = [
        {
            "identifier": "HP:0001251",
            "offset": 50,
            "length": 6,
            "text": "Ataxia",
            "note": None,
        }
    ]
    selected = {"P1": {0}, "P2": {0}}

    assert _filter_selected_indices_by_structure(
        document,
        entities,
        selected,
        previous_distance=100,
        next_distance=0,
    ) == {"P1": {0}}


def test_abbreviation_candidates_propagate_forward_definition() -> None:
    text = "Lattice corneal dystrophy (LCD). LCD progressed."
    long_form = "Lattice corneal dystrophy"
    document = {
        "pmc_id": "1",
        "full_text": [{"section_type": "CASE", "offset": 100, "text": text}],
    }
    base = [
        {
            "identifier": "HP:0001149",
            "offset": 100,
            "length": len(long_form),
            "text": long_form,
            "note": None,
        }
    ]

    candidates = discover_abbreviation_entities(document, base)

    assert [entity["offset"] for entity in candidates] == [
        100 + text.index("LCD"),
        100 + text.rindex("LCD"),
    ]
    assert {entity["identifier"] for entity in candidates} == {"HP:0001149"}


def test_abbreviation_candidates_support_reverse_definition() -> None:
    text = "DM (diabetes mellitus) was diagnosed; DM persisted."
    long_form = "diabetes mellitus"
    document = {
        "pmc_id": "1",
        "full_text": [{"section_type": "CASE", "offset": 0, "text": text}],
    }
    base = [
        {
            "identifier": "HP:0000819",
            "offset": text.index(long_form),
            "length": len(long_form),
            "text": long_form,
            "note": None,
        }
    ]

    candidates = discover_abbreviation_entities(document, base)

    assert [entity["offset"] for entity in candidates] == [
        text.index("DM"),
        text.rindex("DM"),
    ]
    assert {entity["identifier"] for entity in candidates} == {"HP:0000819"}


def test_abbreviation_candidates_drop_ambiguous_definitions() -> None:
    text = "Developmental delay (DD). Distal dysplasia (DD). DD was present."
    first = "Developmental delay"
    second = "Distal dysplasia"
    document = {
        "pmc_id": "1",
        "full_text": [{"section_type": "CASE", "offset": 0, "text": text}],
    }
    base = [
        {
            "identifier": "HP:0001263",
            "offset": text.index(first),
            "length": len(first),
            "text": first,
            "note": None,
        },
        {
            "identifier": "HP:9999999",
            "offset": text.index(second),
            "length": len(second),
            "text": second,
            "note": None,
        },
    ]

    assert discover_abbreviation_entities(document, base) == []
