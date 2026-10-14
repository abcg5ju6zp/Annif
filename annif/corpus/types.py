"""Basic types for document corpora"""

from __future__ import annotations

import abc
import collections
from itertools import islice
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from annif.vocab import SubjectIndex


class Document:
    def __init__(
        self,
        text: str,
        subject_set: SubjectSet | None = None,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
        file_path: str | None = None,
    ):
        self.text = text
        self.subject_set = subject_set if subject_set is not None else SubjectSet()
        self.metadata = metadata if metadata is not None else {}
        self.document_id = document_id
        self.file_path = file_path

    def as_dict(self, subject_index: SubjectIndex, language: str) -> dict[str, Any]:
        doc = {"text": self.text}

        if self.document_id:
            doc["document_id"] = self.document_id
        if self.metadata:
            doc["metadata"] = self.metadata
        if self.subject_set:
            doc["subjects"] = self.subject_set.as_list(subject_index, language)

        return doc

    def __repr__(self):
        return (
            f"Document(document_id={self.document_id!r}, "
            f"text={self.text!r}, "
            f"subject_set={self.subject_set!r}, "
            f"metadata={self.metadata!r}, "
            f"file_path={self.file_path!r})"
        )


class DocumentCorpus(metaclass=abc.ABCMeta):
    """Abstract base class for document corpora"""

    DOC_BATCH_SIZE = 32

    @property
    @abc.abstractmethod
    def documents(self):
        """Iterate through the document corpus, yielding Document objects."""
        pass  # pragma: no cover

    @property
    def doc_batches(self) -> collections.abc.Iterator[list[Document]]:
        """Iterate through the document corpus in batches, yielding lists of Document
        objects."""
        it = iter(self.documents)
        while True:
            docs_batch = list(islice(it, self.DOC_BATCH_SIZE))
            if not docs_batch:
                return
            yield docs_batch

    def is_empty(self) -> bool:
        """Check if there are no documents to iterate."""
        try:
            next(self.documents)
            return False
        except StopIteration:
            return True


class SubjectSet:
    """Represents a set of subjects for a document.

    Besides the accepted (mapped to the project subject space) subject IDs,
    the set carries the evidence of how the gold-standard annotations of the
    document were resolved: the original number of annotations
    (``annotation_count``) and a list of ``(reference, reason)`` pairs for
    every annotation that could not be accepted. This lets all corpus readers
    share the same scorable-evidence determination."""

    def __init__(
        self,
        subject_ids: Any | None = None,
        annotation_count: int | None = None,
        excluded_annotations: list[tuple[str, str]] | None = None,
    ) -> None:
        """Create a SubjectSet and optionally initialize it from an iterable
        of subject IDs. ``annotation_count`` is the number of annotations in
        the original gold standard (accepted and excluded ones counted);
        ``excluded_annotations`` is a list of ``(reference, reason)`` pairs
        for annotations that could not be mapped to the project subject
        space."""

        if subject_ids:
            # use set comprehension to eliminate possible duplicates
            self._subject_ids = list(
                {subject_id for subject_id in subject_ids if subject_id is not None}
            )
        else:
            self._subject_ids = []
        self.annotation_count = (
            annotation_count if annotation_count is not None else len(self._subject_ids)
        )
        self.excluded_annotations = (
            list(excluded_annotations) if excluded_annotations else []
        )

    def __len__(self) -> int:
        return len(self._subject_ids)

    def __getitem__(self, idx: int) -> int:
        return self._subject_ids[idx]

    def __bool__(self) -> bool:
        return bool(self._subject_ids)

    @property
    def has_annotations(self) -> bool:
        """return True if the original gold standard contained at least one
        annotation for this document, even if none of them could be mapped"""

        return self.annotation_count > 0

    @property
    def is_scorable(self) -> bool:
        """return True if the document has at least one gold-standard subject
        that maps to the project subject space and can thus be scored"""

        return bool(self._subject_ids)

    def exclusion_reasons(self) -> collections.Counter:
        """return a Counter of exclusion reasons for the annotations of this
        document"""

        return collections.Counter(reason for _, reason in self.excluded_annotations)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SubjectSet):
            return self._subject_ids == other._subject_ids

        return False

    @classmethod
    def _resolve(
        cls,
        subject_index: SubjectIndex,
        language: str | None,
        uris: Any = (),
        labels: Any = (),
    ) -> SubjectSet:
        """resolve URIs and labels against the subject index, recording every
        accepted and excluded annotation with its exclusion reason"""

        subject_ids = set()
        excluded_annotations = []
        for uri in uris:
            subject_id, reason = subject_index.resolve_uri(uri)
            if subject_id is not None:
                subject_ids.add(subject_id)
            else:
                excluded_annotations.append((uri, reason))
        for label in labels:
            subject_id, reason = subject_index.resolve_label(label, language)
            if subject_id is not None:
                subject_ids.add(subject_id)
            else:
                excluded_annotations.append((str(label), reason))
        annotation_count = len(uris) + len(labels)
        return cls(
            subject_ids,
            annotation_count=annotation_count,
            excluded_annotations=excluded_annotations,
        )

    @classmethod
    def from_uri_references(cls, uris: Any, subject_index: SubjectIndex) -> SubjectSet:
        """create a SubjectSet from an iterable of subject URIs, recording
        unknown, deprecated and filtered URIs as excluded annotations"""

        uri_list = list(uris)
        return cls._resolve(subject_index, language=None, uris=uri_list)

    @classmethod
    def from_subject_dicts(
        cls, subjects: Any, subject_index: SubjectIndex, language: str
    ) -> SubjectSet:
        """create a SubjectSet from an iterable of subject dicts such as the
        ones used by JSON/JSONL corpora, each having either a 'uri' or a
        'label' key"""

        uris = [subj["uri"] for subj in subjects if "uri" in subj]
        labels = [subj["label"] for subj in subjects if "uri" not in subj]
        return cls._resolve(subject_index, language, uris=uris, labels=labels)

    @classmethod
    def from_string(
        cls, subj_data: str, subject_index: SubjectIndex, language: str
    ) -> SubjectSet:
        uris = []
        labels = []
        for line in subj_data.splitlines():
            uri, label = cls._parse_line(line)
            if uri is not None:
                uris.append(uri)
            elif label is not None:
                labels.append(label)
        return cls._resolve(subject_index, language, uris=uris, labels=labels)

    @staticmethod
    def _parse_line(
        line: str,
    ) -> tuple[str | None, str | None]:
        uri = label = None
        vals = line.split("\t")
        for val in vals:
            val = val.strip()
            if val == "":
                continue
            if val.startswith("<") and val.endswith(">"):  # URI
                uri = val[1:-1]
                continue
            label = val
            break
        return uri, label

    def as_vector(
        self, size: int | None = None, destination: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the hits as a one-dimensional NumPy array in sklearn
        multilabel indicator format. Use destination array if given (not
        None), otherwise create and return a new one of the given size."""

        if destination is None:
            import numpy as np

            assert size is not None and size > 0
            destination = np.zeros(size, dtype=bool)

        destination[list(self._subject_ids)] = True

        return destination

    def as_list(
        self, subject_index: SubjectIndex, language: str
    ) -> list[dict[str:str]]:
        subjects = []
        for subject_id in self._subject_ids:
            subject = subject_index[subject_id]
            subjects.append({"uri": subject.uri, "label": subject.labels[language]})
        return subjects

    def __repr__(self):
        return f"SubjectSet({self._subject_ids!r})"
