from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

JsonObject = dict[str, Any]


@dataclass(slots=True)
class Prf:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass(slots=True)
class EvaluationResult:
    mention: Prf
    document: Prf
    association_micro: Prf
    association_macro_precision: float
    association_macro_recall: float
    association_macro_f1: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prf(tp: int, fp: int, fn: int) -> Prf:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Prf(precision, recall, f1, tp, fp, fn)


def _positive_entities(row: JsonObject) -> list[JsonObject]:
    return [entity for entity in row.get("entities", []) if entity.get("note") != "NO"]


def _mention_units(
    row: JsonObject,
) -> tuple[set[tuple[int, int, str]], set[tuple[int, int]], set[tuple[int, int, str]]]:
    regular: set[tuple[int, int, str]] = set()
    no_id: set[tuple[int, int]] = set()
    negative: set[tuple[int, int, str]] = set()
    for entity in row.get("entities", []):
        span = (int(entity["offset"]), int(entity["length"]))
        identifier = str(entity["identifier"])
        if entity.get("note") == "NO":
            for unit in identifier.split(";"):
                negative.add((*span, unit))
            continue
        if identifier == "-1":
            no_id.add(span)
        else:
            for unit in identifier.split(";"):
                regular.add((*span, unit))
    return regular, no_id, negative


def _document_ids(row: JsonObject) -> set[str]:
    identifiers: set[str] = set()
    for entity in _positive_entities(row):
        identifier = str(entity["identifier"])
        if identifier != "-1":
            identifiers.update(identifier.split(";"))
    return identifiers


def evaluate(gold_rows: list[JsonObject], predicted_rows: list[JsonObject]) -> EvaluationResult:
    predictions = {str(row["pmc_id"]): row for row in predicted_rows}
    mention_tp = mention_fp = mention_fn = 0
    document_tp = document_fp = document_fn = 0
    association_tp = association_fp = association_fn = 0
    macro_precision: list[float] = []
    macro_recall: list[float] = []
    macro_f1: list[float] = []

    for gold in gold_rows:
        predicted = predictions.get(
            str(gold["pmc_id"]),
            {"entities": [], "association": []},
        )
        gold_regular, gold_no_id, gold_negative = _mention_units(gold)
        predicted_regular, predicted_no_id, predicted_negative = _mention_units(predicted)

        # Any predicted identifier at a gold -1 span is accepted at mention level.
        predicted_spans = {(offset, length) for offset, length, _ in predicted_regular}
        matched_no_id = gold_no_id.intersection(predicted_no_id | predicted_spans)
        mention_tp += len(gold_regular.intersection(predicted_regular)) + len(matched_no_id)
        mention_fn += len(gold_regular.difference(predicted_regular)) + len(
            gold_no_id.difference(matched_no_id)
        )
        regular_at_no_id = {
            unit for unit in predicted_regular if (unit[0], unit[1]) in gold_no_id
        }
        # An ordinary prediction at a gold-negative span remains in this set and
        # is therefore counted once as an FP, as required by the official rules.
        mention_fp += len(predicted_regular.difference(gold_regular).difference(regular_at_no_id))
        mention_fp += len(predicted_no_id.difference(gold_no_id))
        # Correctly marked negation is ignored; an incorrectly marked negation at a
        # positive span is a missed positive entity, not a predicted positive.
        _ = (gold_negative, predicted_negative)

        gold_ids = _document_ids(gold)
        predicted_ids = _document_ids(predicted)
        document_tp += len(gold_ids.intersection(predicted_ids))
        document_fp += len(predicted_ids.difference(gold_ids))
        document_fn += len(gold_ids.difference(predicted_ids))

        predicted_associations = {
            str(item.get("patient_id")): set(item.get("phenotype", []))
            for item in predicted.get("association", [])
        }
        for gold_association in gold.get("association", []):
            patient_id = str(gold_association["patient_id"])
            gold_set = set(gold_association.get("phenotype", []))
            predicted_set = predicted_associations.get(patient_id, set())
            tp = len(gold_set.intersection(predicted_set))
            fp = len(predicted_set.difference(gold_set))
            fn = len(gold_set.difference(predicted_set))
            association_tp += tp
            association_fp += fp
            association_fn += fn
            patient_prf = _prf(tp, fp, fn)
            macro_precision.append(patient_prf.precision)
            macro_recall.append(patient_prf.recall)
            macro_f1.append(patient_prf.f1)

    mention = _prf(mention_tp, mention_fp, mention_fn)
    document = _prf(document_tp, document_fp, document_fn)
    association_micro = _prf(association_tp, association_fp, association_fn)
    macro_p = sum(macro_precision) / len(macro_precision) if macro_precision else 1.0
    macro_r = sum(macro_recall) / len(macro_recall) if macro_recall else 1.0
    macro_f = sum(macro_f1) / len(macro_f1) if macro_f1 else 1.0
    score = 0.25 * (mention.f1 + document.f1) + 0.25 * (
        association_micro.f1 + macro_f
    )
    return EvaluationResult(
        mention=mention,
        document=document,
        association_micro=association_micro,
        association_macro_precision=macro_p,
        association_macro_recall=macro_r,
        association_macro_f1=macro_f,
        score=score,
    )
