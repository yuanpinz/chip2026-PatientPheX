"""PatientPheX competition solver."""

from .association import (
    associate_by_proximity,
    associate_joint_structured_with_llm,
    associate_joint_with_llm,
    associate_patient_structured_with_llm,
    associate_with_llm,
    propagate_explicit_group_associations,
)
from .cnn_fusion import CnnFusionConfig, cnn_additions, fuse_cnn_entities
from .entities import ExtractorConfig, GazetteerExtractor, merge_entities
from .evaluation import evaluate
from .fusion import fuse_associations_by_vote
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
    "cnn_additions",
    "discover_entities_article_with_llm",
    "discover_entities_with_llm",
    "discover_patient_phenotypes_with_llm",
    "evaluate",
    "fuse_associations_by_vote",
    "fuse_cnn_entities",
    "merge_entities",
    "propagate_explicit_group_associations",
    "read_jsonl",
    "validate_submission",
    "write_jsonl",
]


def main() -> None:
    from .cli import main as cli_main

    cli_main()
