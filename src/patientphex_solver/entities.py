from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ahocorasick

from .ontology import HpoOntology, normalize_surface

JsonObject = dict[str, Any]
_ALPHANUMERIC_RE = re.compile(r"[0-9a-z]", re.IGNORECASE)
_NEGATION_RE = re.compile(
    r"(?:\bno\b|\bnot\b|\bwithout\b|\bden(?:y|ied|ies)\b|"
    r"\bnegative for\b|\bfree of\b|\babsence of\b|\babsent\b|"
    r"\bneither\b|\bnever\b)[^.;:!?]{0,55}$",
    re.IGNORECASE,
)
_GENERIC_HPO_ALIASES = {
    "abnormal",
    "abnormality",
    "acute",
    "all",
    "bilateral",
    "clinical abnormality",
    "congenital",
    "decreased",
    "disorder",
    "generalized",
    "high",
    "increased",
    "large",
    "low",
    "mild",
    "moderate",
    "phenotypic abnormality",
    "severe",
    "short",
    "small",
    "unilateral",
    "wide",
}


def _match_normalize(text: str) -> str:
    # NFKC/case-fold preserves offsets for the English PMC text used here in practice.
    return normalize_surface(text, collapse_space=False)


def _valid_boundary(text: str, start: int, end: int) -> bool:
    if start and text[start - 1].isalnum() and text[start].isalnum():
        return False
    return not (end < len(text) and text[end - 1].isalnum() and text[end].isalnum())


@dataclass(slots=True)
class SurfaceStats:
    identifier_counts: Counter[str] = field(default_factory=Counter)
    positive_labels: int = 0
    negative_labels: int = 0
    occurrences: int = 0

    @property
    def labels(self) -> int:
        return self.positive_labels + self.negative_labels

    @property
    def precision(self) -> float:
        if not self.occurrences:
            return 0.0
        return min(1.0, self.labels / self.occurrences)


@dataclass(slots=True)
class ExtractorConfig:
    train_min_precision: float = 0.35
    hpo_min_chars: int = 4
    hpo_min_alpha: int = 2
    include_hpo_synonyms: bool = True
    include_negated: bool = True
    # The old PhenoTagger dictionary is useful for experiments, but its aliases
    # contain ordinary English words. Keep it opt-in so the default submission
    # does not trade precision for negligible cross-validation recall gains.
    phenotagger_dictionary: str | None = None


class GazetteerExtractor:
    def __init__(
        self,
        ontology: HpoOntology,
        training_documents: Iterable[JsonObject],
        config: ExtractorConfig | None = None,
    ) -> None:
        self.ontology = ontology
        self.training_documents = list(training_documents)
        self.config = config or ExtractorConfig()
        self.id_frequency: Counter[str] = Counter()
        self.surface_stats = self._build_surface_stats()
        self.alias_identifiers: dict[str, set[str]] = defaultdict(set)
        self.alias_source: dict[str, str] = {}
        self._build_aliases()
        self.automaton = self._build_automaton(self.alias_identifiers)

    @staticmethod
    def _build_automaton(aliases: Iterable[str]) -> ahocorasick.Automaton:
        automaton = ahocorasick.Automaton()
        for alias in aliases:
            if alias:
                automaton.add_word(alias, alias)
        automaton.make_automaton()
        return automaton

    def _build_surface_stats(self) -> dict[str, SurfaceStats]:
        stats: dict[str, SurfaceStats] = defaultdict(SurfaceStats)
        aliases: set[str] = set()
        annotated_spans: set[tuple[str, int, int, str]] = set()
        for document in self.training_documents:
            pmc_id = str(document["pmc_id"])
            for entity in document.get("entities", []):
                alias = _match_normalize(str(entity["text"]))
                aliases.add(alias)
                item = stats[alias]
                identifier = str(entity["identifier"])
                item.identifier_counts[identifier] += 1
                for unit in identifier.split(";"):
                    if unit.startswith("HP:"):
                        self.id_frequency[self.ontology.canonical_id(unit)] += 1
                unique_key = (
                    pmc_id,
                    int(entity["offset"]),
                    int(entity["length"]),
                    alias,
                )
                if unique_key not in annotated_spans:
                    if entity.get("note") == "NO":
                        item.negative_labels += 1
                    else:
                        item.positive_labels += 1
                    annotated_spans.add(unique_key)

        if not aliases:
            return dict(stats)
        automaton = self._build_automaton(aliases)
        for document in self.training_documents:
            for passage in document.get("full_text", []):
                normalized_text = _match_normalize(passage.get("text", ""))
                seen: set[tuple[int, int, str]] = set()
                for end_index, alias in automaton.iter(normalized_text):
                    start = end_index - len(alias) + 1
                    end = end_index + 1
                    if _valid_boundary(normalized_text, start, end):
                        seen.add((start, end, alias))
                for _, _, alias in seen:
                    stats[alias].occurrences += 1
        return dict(stats)

    def _build_aliases(self) -> None:
        for alias, stats in self.surface_stats.items():
            if (
                stats.labels
                and stats.precision >= self.config.train_min_precision
                and _ALPHANUMERIC_RE.search(alias)
            ):
                self.alias_identifiers[alias].update(stats.identifier_counts)
                self.alias_source[alias] = "train"

        for alias, identifiers in self.ontology.aliases.items():
            if alias in _GENERIC_HPO_ALIASES:
                continue
            if len(alias) < self.config.hpo_min_chars:
                continue
            if sum(character.isalpha() for character in alias) < self.config.hpo_min_alpha:
                continue
            observed = self.surface_stats.get(alias)
            if observed is not None and observed.occurrences and (
                observed.precision < self.config.train_min_precision
            ):
                continue
            if not self.config.include_hpo_synonyms and alias not in self.ontology.preferred_aliases:
                continue
            self.alias_identifiers[alias].update(identifiers)
            self.alias_source.setdefault(alias, "hpo")

        dictionary_path = self.config.phenotagger_dictionary
        if dictionary_path:
            try:
                external = json.loads(
                    Path(dictionary_path).read_text(encoding="utf-8")
                )
            except (OSError, TypeError, ValueError):
                external = {}
            if isinstance(external, dict):
                for raw_alias, raw_identifiers in external.items():
                    if not isinstance(raw_alias, str) or not isinstance(raw_identifiers, list):
                        continue
                    alias = _match_normalize(raw_alias.replace("_", " "))
                    if len(alias) < self.config.hpo_min_chars:
                        continue
                    if sum(character.isalpha() for character in alias) < self.config.hpo_min_alpha:
                        continue
                    mapped = {
                        self.ontology.canonical_id(str(identifier))
                        for identifier in raw_identifiers
                        if self.ontology.canonical_id(str(identifier)) in self.ontology.descendants
                    }
                    if not mapped:
                        continue
                    observed = self.surface_stats.get(alias)
                    if observed is not None and observed.occurrences and (
                        observed.precision < self.config.train_min_precision
                    ):
                        continue
                    self.alias_identifiers[alias].update(mapped)
                    self.alias_source.setdefault(alias, "phenotagger")

    def _identifier_for_alias(self, alias: str) -> str | None:
        stats = self.surface_stats.get(alias)
        if stats and stats.identifier_counts:
            identifier = stats.identifier_counts.most_common(1)[0][0]
            if identifier != "-1":
                return ";".join(
                    self.ontology.canonical_id(unit) for unit in identifier.split(";")
                )
            return identifier
        candidates = self.alias_identifiers.get(alias, set())
        if not candidates:
            return None
        preferred = self.ontology.preferred_aliases.get(alias, set()).intersection(candidates)
        pool = preferred or candidates
        return max(
            pool,
            key=lambda identifier: (self.id_frequency.get(identifier, 0), identifier),
        )

    @staticmethod
    def _is_negated(text: str, start: int) -> bool:
        sentence_start = max(
            text.rfind(".", 0, start),
            text.rfind(";", 0, start),
            text.rfind("!", 0, start),
            text.rfind("?", 0, start),
            text.rfind("\n", 0, start),
        )
        prefix = text[sentence_start + 1 : start]
        return bool(_NEGATION_RE.search(prefix))

    def extract_document(self, document: JsonObject) -> list[JsonObject]:
        entities: list[JsonObject] = []
        seen: set[tuple[int, int, str, str | None]] = set()
        for passage in document.get("full_text", []):
            text = str(passage.get("text", ""))
            normalized_text = _match_normalize(text)
            passage_offset = int(passage["offset"])
            for end_index, alias in self.automaton.iter(normalized_text):
                start = end_index - len(alias) + 1
                end = end_index + 1
                if not _valid_boundary(normalized_text, start, end):
                    continue
                identifier = self._identifier_for_alias(alias)
                if identifier is None or identifier == self.ontology.root:
                    continue
                note = "NO" if self._is_negated(text, start) else None
                if note == "NO" and not self.config.include_negated:
                    continue
                entity = {
                    "identifier": identifier,
                    "type": "Phenotype",
                    "offset": passage_offset + start,
                    "length": end - start,
                    "text": text[start:end],
                    "note": note,
                }
                key = (
                    entity["offset"],
                    entity["length"],
                    entity["identifier"],
                    entity["note"],
                )
                if key not in seen:
                    seen.add(key)
                    entities.append(entity)
        return sorted(entities, key=lambda item: (item["offset"], -item["length"], item["identifier"]))


def merge_entities(*collections: Iterable[JsonObject]) -> list[JsonObject]:
    by_key: dict[tuple[int, int, str, str | None], JsonObject] = {}
    for collection in collections:
        for entity in collection:
            key = (
                int(entity["offset"]),
                int(entity["length"]),
                str(entity["identifier"]),
                entity.get("note"),
            )
            by_key[key] = entity
    return sorted(
        by_key.values(),
        key=lambda item: (int(item["offset"]), -int(item["length"]), str(item["identifier"])),
    )
