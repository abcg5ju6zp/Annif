"""Shared scorable-evidence determination for gold-standard corpora.

All evaluation entry points (TSV, CSV, JSONL and directory corpora, in both
serial and parallel processing) use GoldCorpusCoverage to decide which
documents can be scored. Documents fall into three disjoint classes:

* scorable documents: at least one gold-standard annotation maps to the
  project subject space,
* documents with only unscorable annotations: the gold standard contains
  annotations, but none of them map (unknown URIs, deprecated subjects or
  subjects excluded by the project filter rules),
* documents without annotations: the gold standard contains no annotations
  at all.

Only scorable documents may enter aggregated metrics, per-subject results,
threshold optimization and backend hyperparameter optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from annif.exception import NotSupportedException
from annif.vocab.types import EXCLUSION_REASONS

from .types import DocumentCorpus

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annif.corpus.types import Document

# the three document classes
DOC_SCORABLE = "scorable"
DOC_UNSCORABLE = "unscorable_annotations"
DOC_WITHOUT_ANNOTATIONS = "no_annotations"


class ScorableDocumentCorpus(DocumentCorpus):
    """A document corpus wrapper that yields only the documents which have at
    least one scorable gold-standard subject. It shares the scorable-evidence
    determination with GoldCorpusCoverage, so serial and parallel processing
    always operate on the same document set and order."""

    def __init__(self, corpus: DocumentCorpus) -> None:
        self._orig_corpus = corpus

    @property
    def documents(self) -> Iterator[Document]:
        for doc in self._orig_corpus.documents:
            if doc.subject_set.is_scorable:
                yield doc


def _document_identity(doc: Document, ordinal: int) -> str:
    if doc.document_id:
        return str(doc.document_id)
    if doc.file_path:
        return doc.file_path
    return f"document #{ordinal}"


class GoldCorpusCoverage:
    """Coverage statistics describing how the gold-standard annotations of a
    corpus resolve against the project subject space."""

    def __init__(self) -> None:
        self.documents_seen = 0
        self.scored_documents = 0
        self.unscorable_documents = 0
        self.without_annotation_documents = 0
        self.annotations_total = 0
        self.accepted_annotations = 0
        self.excluded_annotations = 0
        self.excluded_annotation_reasons = {reason: 0 for reason in EXCLUSION_REASONS}
        # one record per excluded document:
        # (identity, document_class, annotation_count, reasons_counter)
        self.excluded_document_records: list[tuple[str, str, int, dict]] = []

    def add(self, doc: Document) -> None:
        """record the evidence of one document"""

        self.documents_seen += 1
        subject_set = doc.subject_set
        self.annotations_total += subject_set.annotation_count
        self.accepted_annotations += len(subject_set)

        reasons = subject_set.exclusion_reasons()
        for reason, count in reasons.items():
            self.excluded_annotation_reasons[reason] = (
                self.excluded_annotation_reasons.get(reason, 0) + count
            )
        self.excluded_annotations += len(subject_set.excluded_annotations)

        if subject_set.is_scorable:
            self.scored_documents += 1
            return

        if subject_set.has_annotations:
            doc_class = DOC_UNSCORABLE
            self.unscorable_documents += 1
        else:
            doc_class = DOC_WITHOUT_ANNOTATIONS
            self.without_annotation_documents += 1
        self.excluded_document_records.append(
            (
                _document_identity(doc, self.documents_seen),
                doc_class,
                subject_set.annotation_count,
                dict(reasons),
            )
        )

    @classmethod
    def from_corpus(cls, corpus: DocumentCorpus) -> GoldCorpusCoverage:
        """iterate through the corpus once and collect its coverage
        statistics"""

        coverage = cls()
        for doc in corpus.documents:
            coverage.add(doc)
        return coverage

    @property
    def excluded_documents(self) -> int:
        """number of documents that cannot be scored"""

        return self.unscorable_documents + self.without_annotation_documents

    def require_scorable_documents(self) -> None:
        """raise an exception if the corpus contains no scorable documents.
        This is called before any metric calculation or hyperparameter trial
        is started."""

        if self.scored_documents > 0:
            return
        if self.documents_seen == 0:
            raise NotSupportedException("cannot evaluate empty corpus")
        details = (
            f"{self.unscorable_documents} document(s) with only unscorable "
            f"annotations and {self.without_annotation_documents} document(s) "
            "without annotations"
        )
        raise NotSupportedException(
            "cannot evaluate corpus: no documents with a scorable gold "
            f"standard ({details}; {self.excluded_annotations} annotation(s) "
            f"excluded: {self.excluded_annotation_reasons})"
        )

    def excluded_document_dicts(self) -> list[dict]:
        """machine-readable records of the excluded documents, sorted by
        identity so that serial and parallel runs give identical output"""

        records = [
            {
                "document_id": identity,
                "class": doc_class,
                "annotations": annotation_count,
                "excluded_annotation_reasons": dict(reasons),
            }
            for identity, doc_class, annotation_count, reasons in sorted(
                self.excluded_document_records
            )
        ]
        return records

    def as_dict(self) -> dict:
        """machine-readable coverage statistics for the metrics JSON file.
        The number of scored documents is reported separately as the
        'Documents_evaluated' metric."""

        return {
            "documents_excluded": self.excluded_documents,
            "documents_without_annotations": self.without_annotation_documents,
            "documents_with_unscorable_annotations": self.unscorable_documents,
            "annotations_total": self.annotations_total,
            "annotations_accepted": self.accepted_annotations,
            "annotations_excluded": self.excluded_annotations,
            "excluded_annotations_by_reason": dict(self.excluded_annotation_reasons),
            "excluded_documents": self.excluded_document_dicts(),
        }

    def cli_lines(self) -> list[tuple[str, int]]:
        """human-readable coverage lines (label, value) for the command-line
        summary; the number of evaluated documents is reported separately as
        part of the metrics, so it is not repeated here"""

        return [
            ("Documents excluded", self.excluded_documents),
            ("Documents without annotations", self.without_annotation_documents),
            (
                "Documents with only unscorable annotations",
                self.unscorable_documents,
            ),
            ("Annotations excluded", self.excluded_annotations),
            *[
                (f"Annotations excluded ({reason})", self.excluded_annotation_reasons[reason])
                for reason in EXCLUSION_REASONS
            ],
        ]

    def write_results_file_footer(self, results_file) -> None:
        """append the coverage statistics to a per-subject results file as
        comment lines"""

        print("# Gold standard coverage", file=results_file)
        print(f"# Documents evaluated:\t{self.scored_documents}", file=results_file)
        for label, value in self.cli_lines():
            print(f"# {label}:\t{value}", file=results_file)
        for identity, doc_class, _annotation_count, _reasons in sorted(
            self.excluded_document_records
        ):
            print(f"# Excluded document:\t{identity}\t{doc_class}", file=results_file)
