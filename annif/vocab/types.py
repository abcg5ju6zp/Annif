"""Type declarations for vocabulary functionality"""

from __future__ import annotations

import abc
import collections

Subject = collections.namedtuple("Subject", "uri labels notation")

# Reasons why a gold-standard annotation cannot be mapped to the project
# subject space. These are stable machine-readable codes shared by all
# corpus readers and evaluation entry points.
REASON_UNKNOWN_URI = "unknown_uri"
REASON_DEPRECATED_SUBJECT = "deprecated_subject"
REASON_EXCLUDED_SUBJECT = "excluded_subject"
REASON_UNKNOWN_LABEL = "unknown_label"

EXCLUSION_REASONS = (
    REASON_UNKNOWN_URI,
    REASON_DEPRECATED_SUBJECT,
    REASON_EXCLUDED_SUBJECT,
    REASON_UNKNOWN_LABEL,
)


class VocabSource(metaclass=abc.ABCMeta):
    """Abstract base class for vocabulary sources"""

    @property
    @abc.abstractmethod
    def subjects(self):
        """Iterate through the vocabulary, yielding Subject objects."""
        pass  # pragma: no cover

    @property
    @abc.abstractmethod
    def languages(self):
        """Provide a list of language codes supported by this vocabulary source."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def save_skos(self, path):
        """Save the contents of the vocabulary source into a SKOS/Turtle
        file with the given path name."""
        pass  # pragma: no cover


class SubjectIndex(metaclass=abc.ABCMeta):
    """Base class for an index that remembers the associations between
    integer subject IDs and their URIs and labels."""

    @abc.abstractmethod
    def __len__(self) -> int:
        pass  # pragma: no cover

    @property
    @abc.abstractmethod
    def languages(self) -> list[str] | None:
        pass  # pragma: no cover

    @abc.abstractmethod
    def __getitem__(self, subject_id: int) -> Subject:
        pass  # pragma: no cover

    @abc.abstractmethod
    def contains_uri(self, uri: str) -> bool:
        pass  # pragma: no cover

    @abc.abstractmethod
    def by_uri(self, uri: str, warnings: bool = True) -> int | None:
        """return the subject ID of a subject by its URI, or None if not found.
        If warnings=True, log a warning message if the URI cannot be found."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def resolve_uri(
        self, uri: str, warnings: bool = True
    ) -> tuple[int | None, str | None]:
        """resolve a subject URI to its subject ID. Returns a tuple of the
        subject ID and None when the URI is usable, or None and a stable
        reason code (see the REASON_* constants) when it is not. If
        warnings=True, log a warning message when the URI is unknown."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def by_label(self, label: str | None, language: str) -> int | None:
        """return the subject ID of a subject by its label in a given
        language"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def resolve_label(
        self, label: str | None, language: str, warnings: bool = True
    ) -> tuple[int | None, str | None]:
        """resolve a subject label in the given language to its subject ID.
        Returns a tuple of the subject ID and None when the label is usable,
        or None and a stable reason code (see the REASON_* constants) when it
        is not. If warnings=True, log a warning message when the label is
        unknown."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def active(self) -> list[tuple[int, Subject]]:
        """return a list of (subject_id, Subject) tuples of all subjects that
        are available for use"""
        pass  # pragma: no cover
