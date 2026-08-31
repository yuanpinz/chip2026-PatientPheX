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
from .association_judge import (
    associate_values_calibrated_with_llm,
    associate_values_joint_calibrated_with_llm,
    build_association_calibration_examples,
)
from .cnn_fusion import CnnFusionConfig, fuse_cnn_entities
from .entities import ExtractorConfig, GazetteerExtractor, merge_entities
from .entity_judge import build_calibration_examples, judge_entities_with_llm
from .evaluation import evaluate
from .fusion import fuse_associations_by_patient_count
from .io import read_jsonl, validate_submission, write_jsonl
from .llm import BigModelClient
from .llm_entities import (
    discover_entities_article_with_llm,
    discover_entities_fewshot_with_llm,
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


def _cmd_fuse_associations(args: argparse.Namespace) -> None:
    expected = read_jsonl(args.expected)
    predictions = fuse_associations_by_patient_count(
        expected,
        read_jsonl(args.base),
        read_jsonl(args.primary),
        read_jsonl(args.secondary),
        union_multi=args.union_multi,
        structure_previous_distance=args.structure_previous_distance,
        structure_next_distance=args.structure_next_distance,
    )
    errors = validate_submission(predictions, expected)
    if errors:
        raise SystemExit("submission validation failed:\n" + "\n".join(errors[:30]))
    write_jsonl(args.output, predictions)
    print(f"wrote {args.output} ({len(predictions)} documents)")


def _cmd_fuse_cnn_entities(args: argparse.Namespace) -> None:
    base_rows = read_jsonl(args.base)
    predictions = fuse_cnn_entities(
        base_rows,
        read_jsonl(args.cnn),
        CnnFusionConfig(
            min_score=args.min_score,
            min_text_length=args.min_text_length,
            max_per_identifier=args.max_per_identifier,
            avoid_existing_overlap=not args.allow_overlap,
            new_identifiers_only=args.new_identifiers_only,
        ),
    )
    write_jsonl(args.output, predictions)
    additions = sum(
        max(0, len(row.get("entities", [])) - len(base.get("entities", [])))
        for row, base in zip(predictions, base_rows)
    )
    if args.additions_output:
        additions_rows = []
        for row, base in zip(predictions, base_rows):
            additions_rows.append(
                {
                    "pmc_id": row["pmc_id"],
                    "pmid": row.get("pmid"),
                    "entities": list(row.get("entities", []))[len(base.get("entities", [])) :],
                    "association": [],
                }
            )
        write_jsonl(args.additions_output, additions_rows)
    print(f"wrote {args.output} ({len(predictions)} documents, {additions} CNN additions)")


def _cmd_merge_entities(args: argparse.Namespace) -> None:
    base_rows = read_jsonl(args.base)
    additions_by_id = {
        str(row["pmc_id"]): row for row in read_jsonl(args.additions)
    }
    predictions = []
    for base in base_rows:
        additions = additions_by_id.get(str(base["pmc_id"]))
        if additions is None:
            raise ValueError(f"missing additions row for PMC {base['pmc_id']}")
        predictions.append(
            {
                "pmc_id": base["pmc_id"],
                "pmid": base.get("pmid"),
                "entities": merge_entities(
                    list(base.get("entities", [])),
                    list(additions.get("entities", [])),
                ),
                "association": list(base.get("association", [])),
            }
        )
    write_jsonl(args.output, predictions)
    print(f"wrote {args.output} ({len(predictions)} documents)")


def _cmd_judge_entities(args: argparse.Namespace) -> None:
    train_path, test_path, ontology_path = _paths(args)
    train = read_jsonl(train_path)
    documents = read_jsonl(test_path if args.split == "a" else train_path)
    documents_by_id = {str(row["pmc_id"]): row for row in documents}
    candidates = read_jsonl(args.candidates)
    ontology = HpoOntology.from_obo(ontology_path)
    extractor = GazetteerExtractor(ontology, train)
    calibration = build_calibration_examples(train, extractor, ontology)
    client = BigModelClient(model=args.model, cache_dir=args.cache_dir)
    predictions: list[dict] = []
    progress_path = str(args.output) + ".progress"
    completed = {
        str(row["pmc_id"]): row
        for row in read_jsonl(progress_path)
    } if Path(progress_path).exists() else {}
    for index, candidate_row in enumerate(candidates, 1):
        pmc_id = str(candidate_row["pmc_id"])
        if pmc_id in completed:
            predictions.append(completed[pmc_id])
            print(f"[{index}/{len(candidates)}] {pmc_id} (cached)", flush=True)
            continue
        document = documents_by_id.get(pmc_id)
        if document is None:
            raise ValueError(f"candidate document {pmc_id} is not in split {args.split}")
        # During train probes, do not let the model see calibration labels from
        # the article it is currently judging.
        document_calibration = calibration
        if args.split == "train":
            other_documents = [
                row for row in train if str(row["pmc_id"]) != pmc_id
            ]
            other_extractor = GazetteerExtractor(ontology, other_documents)
            document_calibration = build_calibration_examples(
                other_documents, other_extractor, ontology
            )
        print(
            f"[{index}/{len(candidates)}] {pmc_id} "
            f"({len(candidate_row.get('entities', []))} candidates)",
            flush=True,
        )
        entities = judge_entities_with_llm(
            document,
            list(candidate_row.get("entities", [])),
            ontology,
            client,
            document_calibration,
            batch_size=args.batch_size,
            calibration_per_label=args.calibration_per_label,
            include_uncertain=args.include_uncertain,
        )
        association = candidate_row.get("association")
        if not isinstance(association, list):
            association = associate_by_proximity(document, entities)
        predictions.append(
            {
                "pmc_id": document["pmc_id"],
                "pmid": document.get("pmid"),
                "entities": entities,
                "association": association,
            }
        )
        write_jsonl(progress_path, predictions)
    write_jsonl(args.output, predictions)
    print(f"wrote {args.output} ({len(predictions)} documents)")


def _cmd_judge_associations(args: argparse.Namespace) -> None:
    train_path, test_path, ontology_path = _paths(args)
    train = read_jsonl(train_path)
    documents = read_jsonl(test_path if args.split == "a" else train_path)
    documents_by_id = {str(row["pmc_id"]): row for row in documents}
    candidates = read_jsonl(args.candidates)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    ontology = HpoOntology.from_obo(ontology_path)
    calibration = build_association_calibration_examples(train, ontology)
    client = BigModelClient(model=args.model, cache_dir=args.cache_dir)
    predictions: list[dict] = []
    progress_path = str(args.output) + ".progress"
    completed = {
        str(row["pmc_id"]): row
        for row in read_jsonl(progress_path)
    } if Path(progress_path).exists() else {}
    for index, candidate_row in enumerate(candidates, 1):
        pmc_id = str(candidate_row["pmc_id"])
        if pmc_id in completed:
            predictions.append(completed[pmc_id])
            print(f"[{index}/{len(candidates)}] {pmc_id} (cached)", flush=True)
            continue
        document = documents_by_id.get(pmc_id)
        if document is None:
            raise ValueError(f"candidate document {pmc_id} is not in split {args.split}")
        entities = list(candidate_row.get("entities", []))
        print(
            f"[{index}/{len(candidates)}] {pmc_id} "
            f"({len(entities)} entities, {len(document.get('patient', []))} patients)",
            flush=True,
        )
        association = (
            associate_values_joint_calibrated_with_llm
            if args.joint
            else associate_values_calibrated_with_llm
        )(
            document,
            entities,
            ontology,
            client,
            calibration,
            batch_size=args.batch_size,
            calibration_per_label=args.calibration_per_label,
            include_uncertain=args.include_uncertain,
            exclude_calibration_pmc_id=pmc_id if args.split == "train" else None,
            structure_multi_patient=not args.no_structure_filter,
        )
        predictions.append(
            {
                "pmc_id": document["pmc_id"],
                "pmid": document.get("pmid"),
                "entities": entities,
                "association": association,
            }
        )
        write_jsonl(progress_path, predictions)
    write_jsonl(args.output, predictions)
    print(f"wrote {args.output} ({len(predictions)} documents)")


def _cmd_discover_entities(args: argparse.Namespace) -> None:
    train_path, test_path, ontology_path = _paths(args)
    train = read_jsonl(train_path)
    documents = read_jsonl(test_path if args.split == "a" else train_path)
    documents_by_id = {str(row["pmc_id"]): row for row in documents}
    candidates = read_jsonl(args.candidates)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    ontology = HpoOntology.from_obo(ontology_path)
    extractor = GazetteerExtractor(ontology, train)
    client = BigModelClient(model=args.model, cache_dir=args.cache_dir)
    predictions: list[dict] = []
    progress_path = str(args.output) + ".progress"
    completed = {
        str(row["pmc_id"]): row
        for row in read_jsonl(progress_path)
    } if Path(progress_path).exists() else {}
    for index, candidate_row in enumerate(candidates, 1):
        pmc_id = str(candidate_row["pmc_id"])
        if pmc_id in completed:
            predictions.append(completed[pmc_id])
            print(f"[{index}/{len(candidates)}] {pmc_id} (cached)", flush=True)
            continue
        document = documents_by_id.get(pmc_id)
        if document is None:
            raise ValueError(f"candidate document {pmc_id} is not in split {args.split}")
        known = list(candidate_row.get("entities", []))
        examples = [row for row in train if str(row["pmc_id"]) != pmc_id]
        print(
            f"[{index}/{len(candidates)}] {pmc_id} ({len(known)} known entities)",
            flush=True,
        )
        additions = discover_entities_fewshot_with_llm(
            document,
            known,
            examples,
            ontology,
            client,
            id_frequency=extractor.id_frequency,
            max_chars=args.max_chars,
            max_examples=args.max_examples,
        )
        predictions.append(
            {
                "pmc_id": document["pmc_id"],
                "pmid": document.get("pmid"),
                "base": known,
                "additions": additions,
                "entities": merge_entities(known, additions),
                "association": candidate_row.get("association", []),
            }
        )
        write_jsonl(progress_path, predictions)
    write_jsonl(args.output, predictions)
    print(f"wrote {args.output} ({len(predictions)} documents)")


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

    fuse = subparsers.add_parser(
        "fuse-associations",
        help="union single-patient predictions and use joint multi-patient predictions",
    )
    fuse.add_argument("--expected", required=True)
    fuse.add_argument("--base", required=True, help="JSONL providing final entities")
    fuse.add_argument("--primary", required=True, help="primary association JSONL")
    fuse.add_argument("--secondary", required=True, help="joint association JSONL")
    fuse.add_argument("--output", required=True)
    fuse.add_argument(
        "--union-multi",
        action="store_true",
        help="union primary and secondary associations for multi-patient articles",
    )
    fuse.add_argument(
        "--structure-previous-distance",
        type=int,
        default=None,
        help="optional local structure filter distance before an entity (characters)",
    )
    fuse.add_argument(
        "--structure-next-distance",
        type=int,
        default=None,
        help="optional local structure filter distance after an entity (characters)",
    )
    fuse.set_defaults(func=_cmd_fuse_associations)

    fuse_cnn = subparsers.add_parser(
        "fuse-cnn-entities",
        help="add conservative high-confidence PhenoTagger CNN entities",
    )
    fuse_cnn.add_argument("--base", required=True, help="base entity JSONL")
    fuse_cnn.add_argument("--cnn", required=True, help="raw PhenoTagger CNN JSONL")
    fuse_cnn.add_argument("--output", required=True)
    fuse_cnn.add_argument(
        "--additions-output",
        default=None,
        help="optional JSONL containing only CNN additions for API judging",
    )
    fuse_cnn.add_argument("--min-score", type=float, default=0.9997)
    fuse_cnn.add_argument("--min-text-length", type=int, default=6)
    fuse_cnn.add_argument("--max-per-identifier", type=int, default=10)
    fuse_cnn.add_argument(
        "--allow-overlap",
        action="store_true",
        help="allow CNN spans overlapping existing entities",
    )
    fuse_cnn.add_argument(
        "--new-identifiers-only",
        action="store_true",
        help="only add HPO IDs absent from the base document",
    )
    fuse_cnn.set_defaults(func=_cmd_fuse_cnn_entities)

    merge = subparsers.add_parser(
        "merge-entities",
        help="merge an additions JSONL into a base entity JSONL",
    )
    merge.add_argument("--base", required=True)
    merge.add_argument("--additions", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=_cmd_merge_entities)

    judge = subparsers.add_parser(
        "judge-entities",
        help="filter an entity candidate JSONL with a calibrated API model",
    )
    judge.add_argument("--data-dir", default="PatientPheX-V1-A")
    judge.add_argument("--split", choices=["a", "train"], default="a")
    judge.add_argument("--candidates", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--model", default="modelH")
    judge.add_argument("--cache-dir", default="cache/llm")
    judge.add_argument("--batch-size", type=int, default=40)
    judge.add_argument("--calibration-per-label", type=int, default=10)
    judge.add_argument("--include-uncertain", action="store_true")
    judge.set_defaults(func=_cmd_judge_entities)

    judge_associations = subparsers.add_parser(
        "judge-associations",
        help="assign candidate phenotype values to patients with a calibrated API model",
    )
    judge_associations.add_argument("--data-dir", default="PatientPheX-V1-A")
    judge_associations.add_argument("--split", choices=["a", "train"], default="a")
    judge_associations.add_argument("--candidates", required=True)
    judge_associations.add_argument("--output", required=True)
    judge_associations.add_argument("--model", default="modelS5_6S")
    judge_associations.add_argument("--cache-dir", default="cache/llm")
    judge_associations.add_argument("--batch-size", type=int, default=30)
    judge_associations.add_argument("--calibration-per-label", type=int, default=8)
    judge_associations.add_argument("--include-uncertain", action="store_true")
    judge_associations.add_argument("--no-structure-filter", action="store_true")
    judge_associations.add_argument("--joint", action="store_true")
    judge_associations.add_argument("--limit", type=int, default=None)
    judge_associations.set_defaults(func=_cmd_judge_associations)

    discover = subparsers.add_parser(
        "discover-entities",
        help="add few-shot passage-level API entities to a candidate JSONL",
    )
    discover.add_argument("--data-dir", default="PatientPheX-V1-A")
    discover.add_argument("--split", choices=["a", "train"], default="a")
    discover.add_argument("--candidates", required=True)
    discover.add_argument("--output", required=True)
    discover.add_argument("--model", default="modelS5_6S")
    discover.add_argument("--cache-dir", default="cache/llm")
    discover.add_argument("--max-chars", type=int, default=6500)
    discover.add_argument("--max-examples", type=int, default=5)
    discover.add_argument("--limit", type=int, default=None)
    discover.set_defaults(func=_cmd_discover_entities)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
