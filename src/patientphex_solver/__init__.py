"""PatientPheX competition solver."""

from .abbreviations import discover_abbreviation_entities
from .association import (
    associate_by_proximity,
    associate_joint_structured_with_llm,
    associate_joint_with_llm,
    associate_patient_structured_with_llm,
    associate_with_llm,
    propagate_explicit_group_associations,
)
from .cnn_fusion import CnnFusionConfig, cnn_additions, fuse_cnn_entities
from .entities import (
    ExtractorConfig,
    GazetteerExtractor,
    merge_entities,
    select_entities_by_vote,
    subtract_entities,
    vote_entities,
)
from .evaluation import evaluate
from .fusion import (
    augment_associations_by_vote,
    clip_associations_to_entities,
    fuse_associations_by_vote,
    stabilize_associations,
)
from .io import read_jsonl, validate_submission, write_jsonl
from .llm import BigModelClient
from .llm_entities import (
    discover_entities_article_with_llm,
    discover_entities_with_llm,
)
from .ontology import HpoOntology
from .patient_phenotypes import discover_patient_phenotypes_with_llm

__all__ = [
    "BigModelClient",
    "CnnFusionConfig",
    "ExtractorConfig",
    "GazetteerExtractor",
    "HpoOntology",
    "associate_by_proximity",
    "associate_joint_structured_with_llm",
    "associate_joint_with_llm",
    "associate_patient_structured_with_llm",
    "associate_with_llm",
    "augment_associations_by_vote",
    "clip_associations_to_entities",
    "cnn_additions",
    "discover_abbreviation_entities",
    "discover_entities_article_with_llm",
    "discover_entities_with_llm",
    "discover_patient_phenotypes_with_llm",
    "evaluate",
    "fuse_associations_by_vote",
    "fuse_cnn_entities",
    "merge_entities",
    "propagate_explicit_group_associations",
    "read_jsonl",
    "select_entities_by_vote",
    "stabilize_associations",
    "subtract_entities",
    "validate_submission",
    "vote_entities",
    "write_jsonl",
]


def main() -> None:
    from .cli import main as cli_main

    cli_main()
