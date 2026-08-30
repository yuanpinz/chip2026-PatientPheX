from __future__ import annotations

import argparse
import json
from pathlib import Path

from .association import (
    associate_by_proximity,
    associate_consensus_structured_with_llm,
    associate_joint_structured_with_llm,
    associate_joint_with_llm,
    associate_patient_structured_with_llm,
    associate_with_llm,
)
from .entities import ExtractorConfig, GazetteerExtractor, merge_entities
from .evaluation import evaluate
from .io import read_jsonl, validate_submission, write_jsonl
from .llm import BigModelClient
from .llm_entities import (
    discover_entities_article_with_llm,
    discover_entities_with_llm,
)
from .ontology import HpoOntology
from .patient_phenotypes import discover_patient_phenotypes_with_llm


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_dir = Path(args.data_dir)
    return (
        data_dir / "PatientPheX-train.jsonl",
        data_dir / "PatientPheX-A.jsonl",
        data_dir / "hp.obo",
    )


def _build_extractor(
    data_dir: str | Path,
    train: list[dict],
    phenotagger_dictionary: str | None = None,
) -> tuple[HpoOntology, GazetteerExtractor]:
    ontology = HpoOntology.from_obo(Path(data_dir) / "hp.obo")
    extractor = GazetteerExtractor(
        ontology,
        train,
        ExtractorConfig(phenotagger_dictionary=phenotagger_dictionary),
    )
    return ontology, extractor


def _predict(
    documents: list[dict],
    ontology: HpoOntology,
    extractor: GazetteerExtractor,
    *,
    use_llm: bool,
    association_mode: str,
    entity_batch: str,
    client: BigModelClient | None,
    progress_path: str | Path | None = None,
) -> list[dict]:
    predictions_by_id: dict[str, dict] = {}
    if progress_path is not None and Path(progress_path).exists():
        for row in read_jsonl(progress_path):
            predictions_by_id[str(row["pmc_id"])] = row
    predictions: list[dict] = []
    for index, document in enumerate(documents, 1):
        if str(document["pmc_id"]) in predictions_by_id:
            predictions.append(predictions_by_id[str(document["pmc_id"])])
            print(
                f"[{index}/{len(documents)}] {document['pmc_id']} (cached)", flush=True
            )
            continue
        print(f"[{index}/{len(documents)}] {document['pmc_id']}", flush=True)
        entities = extractor.extract_document(document)
        direct_association = None
        if association_mode == "patient-direct":
            if client is None:
                raise ValueError("--association patient-direct requires an API client")
            try:
                _, direct_association = discover_patient_phenotypes_with_llm(
                    document, ontology, extractor, client
                )
            except (json.JSONDecodeError, RuntimeError) as exc:
                print(f"  direct patient extraction failed: {exc}", flush=True)
        if use_llm and client is not None:
            discover = (
                discover_entities_article_with_llm
                if entity_batch == "article"
                else discover_entities_with_llm
            )
            additions = discover(
                document,
                entities,
                ontology,
                client,
                id_frequency=extractor.id_frequency,
            )
            entities = merge_entities(entities, additions)
        if direct_association is not None:
            association = direct_association
        elif association_mode in {
            "llm",
            "patient-structured",
            "joint-llm",
            "joint-structured",
            "consensus-structured",
            "joint-intersection",
        }:
            if client is None:
                raise ValueError("--association llm requires an API client")
            try:
                association = (
                    associate_consensus_structured_with_llm(
                        document, entities, client
                    )
                    if association_mode == "consensus-structured"
                    else
                    associate_joint_structured_with_llm(document, entities, client)
                    if association_mode == "joint-structured"
                    else associate_patient_structured_with_llm(
                        document, entities, client
                    )
                    if association_mode == "patient-structured"
                    else associate_joint_with_llm(document, entities, client)
                    if association_mode in {"joint-llm", "joint-intersection"}
                    else associate_with_llm(document, entities, client)
                )
                if association_mode == "joint-intersection":
                    proximity = associate_by_proximity(document, entities)
                    proximity_by_id = {
                        str(item["patient_id"]): set(item.get("phenotype", []))
                        for item in proximity
                    }
                    for item in association:
                        patient_id = str(item["patient_id"])
                        item["phenotype"] = [
                            value
                            for value in item.get("phenotype", [])
                            if value in proximity_by_id.get(patient_id, set())
                        ]
            except RuntimeError as exc:
                print(
                    f"  LLM association failed; using proximity fallback: {exc}",
                    flush=True,
                )
                association = associate_by_proximity(document, entities)
        else:
            association = associate_by_proximity(document, entities)
        predictions.append(
            {
                "pmc_id": document["pmc_id"],
                "pmid": document.get("pmid"),
                "entities": entities,
                "association": association,
            }
        )
        if progress_path is not None:
            write_jsonl(progress_path, predictions)
    return predictions


def _cmd_predict(args: argparse.Namespace) -> None:
    train_path, test_path, _ = _paths(args)
    train = read_jsonl(train_path)
    documents = read_jsonl(test_path if args.split == "a" else train_path)
    ontology, extractor = _build_extractor(
        args.data_dir,
        train,
        args.phenotagger_dictionary,
    )
    client = (
        BigModelClient(model=args.model, cache_dir=args.cache_dir)
        if args.use_llm
        or args.association
        in {
            "llm",
            "patient-structured",
            "joint-llm",
            "joint-structured",
            "consensus-structured",
            "joint-intersection",
            "patient-direct",
        }
        else None
    )
    predictions = _predict(
        documents,
        ontology,
        extractor,
        use_llm=args.use_llm,
        association_mode=args.association,
        entity_batch=args.entity_batch,
        client=client,
        progress_path=str(args.output) + ".progress",
    )
    write_jsonl(args.output, predictions)
    errors = validate_submission(predictions, documents)
    if errors:
        raise SystemExit("submission validation failed:\n" + "\n".join(errors[:30]))
    print(f"wrote {args.output} ({len(predictions)} documents)")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    gold = read_jsonl(args.gold)
    predicted = read_jsonl(args.predicted)
    print(json.dumps(evaluate(gold, predicted).to_dict(), ensure_ascii=False, indent=2))


def _cmd_validate(args: argparse.Namespace) -> None:
    predictions = read_jsonl(args.predicted)
    expected = read_jsonl(args.expected)
    errors = validate_submission(predictions, expected)
    if errors:
        print("INVALID")
        print("\n".join(errors))
        raise SystemExit(1)
    print("VALID")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patientphex-solver")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="generate a JSONL prediction file")
    predict.add_argument("--data-dir", default="PatientPheX-V1-A")
    predict.add_argument("--split", choices=["a", "train"], default="a")
    predict.add_argument("--output", required=True)
    predict.add_argument("--use-llm", action="store_true")
    predict.add_argument(
        "--entity-batch",
        choices=["passage", "article"],
        default="passage",
        help="scope of optional LLM entity discovery calls",
    )
    predict.add_argument(
        "--phenotagger-dictionary",
        default=None,
        help="optional legacy PhenoTagger word_id_map.json",
    )
    predict.add_argument(
        "--association",
        choices=[
            "proximity",
            "llm",
            "patient-structured",
            "joint-llm",
            "joint-structured",
            "consensus-structured",
            "joint-intersection",
            "patient-direct",
        ],
        default="joint-structured",
    )
    predict.add_argument("--model", default="modelK5")
    predict.add_argument("--cache-dir", default="cache/llm")
    predict.set_defaults(func=_cmd_predict)

    scoring = subparsers.add_parser(
        "evaluate", help="evaluate predictions against gold JSONL"
    )
    scoring.add_argument("--gold", required=True)
    scoring.add_argument("--predicted", required=True)
    scoring.set_defaults(func=_cmd_evaluate)

    validate = subparsers.add_parser(
        "validate", help="validate JSONL submission schema"
    )
    validate.add_argument("--expected", required=True)
    validate.add_argument("--predicted", required=True)
    validate.set_defaults(func=_cmd_validate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
