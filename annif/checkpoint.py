"""Resumable checkpoints for batch suggestion and evaluation jobs.

A batch job pins the identity of everything that affects its results:
the project configuration, the vocabulary content, the backend model
version, the backend parameters and the ordered identities of the input
documents. A checkpoint stores the output (suggestions) and the
evaluation aggregate inputs (gold subjects) after every confirmed batch,
so that an interrupted, failed or cancelled job can be resumed without
repeating inference and results from different inputs or model versions
can never be stitched together.

Each run that opens a checkpoint becomes its "owner". Batch outputs
produced for an earlier owner are rejected once a newer owner has taken
over, so late results from a superseded pool cannot cross the new
checkpoint boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import scipy.sparse

import annif

if TYPE_CHECKING:
    from annif.corpus.types import Document
    from annif.suggestion import SuggestionBatch

logger = annif.logger

CHECKPOINT_FORMAT_VERSION = 1

MANIFEST_FILENAME = "manifest.json"

# job types that require the gold subject array to be stored per batch
JOB_TYPES_WITH_GOLD = ("eval",)


class StaleOwnerError(Exception):
    """Raised when a worker result carrying an old owner token is
    committed after a newer owner has taken over the checkpoint."""


def document_identity(doc: Document) -> list:
    """Return a stable, JSON-serializable identity for a document.

    The explicit document ID is preferred, then the file path, and
    finally a SHA-256 hash of the document text."""

    if doc.document_id:
        return ["id", doc.document_id]
    if doc.file_path:
        return ["path", doc.file_path]
    digest = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
    return ["text", digest]


def format_document_identity(identity: list) -> str:
    """Format a document identity (as returned by document_identity)
    for human-readable log messages."""

    kind, value = identity
    if kind == "id":
        return f"document '{value}'"
    if kind == "path":
        return f"file {value}"
    return f"document with text hash {value}"


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def project_identity(project) -> dict[str, Any]:
    """Build a stable identity for a project covering its configuration,
    its vocabulary (content, not just the identifier) and the backend
    model version."""

    config_items = sorted([str(key), str(val)] for key, val in project.config.items())

    subjects = project.subjects
    vocab_hash = hashlib.sha256()
    for subject_id in range(len(subjects)):
        subject = subjects[subject_id]
        labels = tuple(sorted((subject.labels or {}).items()))
        vocab_hash.update(repr((subject.uri, subject.notation, labels)).encode("utf-8"))

    modification_time = project.modification_time
    return {
        "config": config_items,
        "vocab": {
            "spec": project.vocab_spec,
            "language": project.vocab_lang,
            "size": len(subjects),
            "subjects_hash": vocab_hash.hexdigest(),
        },
        "model_version": (
            modification_time.isoformat() if modification_time is not None else None
        ),
    }


@dataclass(frozen=True)
class JobFingerprint:
    """An immutable description of a batch job. Two jobs with the same
    fingerprint operate on the same inputs with the same projects and
    parameters, so their batch outputs are interchangeable. Any change in
    the project configuration, vocabulary, model version, backend
    parameters or input documents produces a different fingerprint."""

    job_type: str
    project_ids: list[str]
    projects: dict[str, dict[str, Any]]
    backend_params: dict[str, dict[str, str]]
    limit: int | None
    threshold: float
    documents: list[list[list]]

    @classmethod
    def build(
        cls,
        job_type: str,
        projects: list,
        backend_params: dict[str, dict[str, str]] | None,
        limit: int | None,
        threshold: float,
        doc_batches: list[list[Document]],
    ) -> JobFingerprint:
        """Create a fingerprint for the given projects and the already
        materialized document batches."""

        project_ids = [proj.project_id for proj in projects]
        return cls(
            job_type=job_type,
            project_ids=list(project_ids),
            projects={proj.project_id: project_identity(proj) for proj in projects},
            backend_params={
                backend_id: dict(params)
                for backend_id, params in sorted((backend_params or {}).items())
            },
            limit=limit,
            threshold=threshold,
            documents=[
                [document_identity(doc) for doc in batch] for batch in doc_batches
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_type": self.job_type,
            "project_ids": list(self.project_ids),
            "projects": self.projects,
            "backend_params": self.backend_params,
            "limit": self.limit,
            "threshold": self.threshold,
            "documents": self.documents,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobFingerprint:
        return cls(
            job_type=data["job_type"],
            project_ids=list(data["project_ids"]),
            projects=dict(data["projects"]),
            backend_params={
                backend_id: dict(params)
                for backend_id, params in data["backend_params"].items()
            },
            limit=data["limit"],
            threshold=data["threshold"],
            documents=[
                [list(identity) for identity in batch] for batch in data["documents"]
            ],
        )

    @property
    def key(self) -> str:
        """Short hash that uniquely identifies this fingerprint."""

        return hashlib.sha256(_stable_json(self.to_dict()).encode("utf-8")).hexdigest()[
            :12
        ]

    @property
    def n_batches(self) -> int:
        return len(self.documents)

    def requires_gold(self) -> bool:
        return self.job_type in JOB_TYPES_WITH_GOLD


class BatchCheckpoint:
    """On-disk checkpoint for a single batch job.

    The checkpoint directory layout is::

        <root>/<job_type>-<fingerprint key>/
            manifest.json
            batch-00000-<project id>.npz
            batch-00000-gold.npz
            ...

    All files are written atomically (temporary file + rename), so a
    crash can never leave a partially written batch behind."""

    def __init__(
        self, root_dir: str, fingerprint: JobFingerprint, manifest: dict[str, Any]
    ) -> None:
        self.root_dir = root_dir
        self.fingerprint = fingerprint
        self.job_dir = os.path.join(
            root_dir, f"{fingerprint.job_type}-{fingerprint.key}"
        )
        self._manifest = manifest

    @classmethod
    def open(cls, root_dir: str, fingerprint: JobFingerprint) -> BatchCheckpoint:
        """Open a checkpoint for the given fingerprint.

        If a checkpoint with the same fingerprint exists, the job resumes
        from it under a new owner token. If the checkpoint exists but the
        fingerprint differs (input documents, project configuration,
        vocabulary or model version changed), the stale checkpoint is
        discarded completely so that old and new results can never be
        mixed."""

        job_dir = os.path.join(root_dir, f"{fingerprint.job_type}-{fingerprint.key}")
        manifest_path = os.path.join(job_dir, MANIFEST_FILENAME)

        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            if manifest.get("fingerprint") == fingerprint.to_dict():
                old_owner = manifest.get("owner")
                manifest["owner"] = uuid.uuid4().hex
                checkpoint = cls(root_dir, fingerprint, manifest)
                checkpoint._write_manifest()
                n_cached = len(manifest.get("batches", {}))
                logger.info(
                    "Resuming %s job from checkpoint %s: %d/%d batches already "
                    "completed (previous owner %s)",
                    fingerprint.job_type,
                    job_dir,
                    n_cached,
                    fingerprint.n_batches,
                    old_owner,
                )
                return checkpoint
            logger.warning(
                "Discarding checkpoint %s: project configuration, vocabulary, "
                "backend parameters, model version or input documents do not "
                "match the current job",
                job_dir,
            )
            shutil.rmtree(job_dir)

        os.makedirs(job_dir, exist_ok=True)
        manifest = {
            "version": CHECKPOINT_FORMAT_VERSION,
            "owner": uuid.uuid4().hex,
            "job_type": fingerprint.job_type,
            "project_ids": list(fingerprint.project_ids),
            "n_batches": fingerprint.n_batches,
            "fingerprint": fingerprint.to_dict(),
            "batches": {},
            "failures": [],
        }
        checkpoint = cls(root_dir, fingerprint, manifest)
        checkpoint._write_manifest()
        return checkpoint

    @property
    def owner_id(self) -> str:
        """Return the token of the current owner of this checkpoint."""

        return self._manifest["owner"]

    def is_owner(self, owner_id: str) -> bool:
        """Return True if the given token belongs to the current owner."""

        return owner_id == self._manifest["owner"]

    @property
    def n_batches(self) -> int:
        return self.fingerprint.n_batches

    @property
    def failures(self) -> list[dict[str, Any]]:
        """Return the records of batches that have failed and still need
        to be retried."""

        return self._manifest["failures"]

    def _manifest_path(self) -> str:
        return os.path.join(self.job_dir, MANIFEST_FILENAME)

    def _read_manifest(self) -> dict[str, Any]:
        """Read the manifest from disk. The disk manifest is the
        authoritative source for the owner token, so a store object held
        by a superseded process cannot commit past a newer checkpoint."""

        with open(self._manifest_path(), encoding="utf-8") as manifest_file:
            return json.load(manifest_file)

    def _write_manifest(self) -> None:
        """Write the manifest atomically."""

        def _write(tmp_path: str) -> None:
            with open(tmp_path, "w", encoding="utf-8") as tmp_file:
                json.dump(self._manifest, tmp_file)

        self._atomic_write(MANIFEST_FILENAME, _write)

    def _check_ownership(self, owner_id: str) -> None:
        disk_owner = self._read_manifest().get("owner")
        if disk_owner != owner_id:
            raise StaleOwnerError(
                f"rejecting commit from superseded owner {owner_id}; "
                f"checkpoint is now owned by {disk_owner}"
            )
        if self._manifest.get("owner") != disk_owner:
            # another owner has taken over in this store's absence
            self._manifest = self._read_manifest()

    def _atomic_write(self, filename: str, write_fn) -> None:
        """Write a file via a temporary file and an atomic rename. The
        temporary file carries the owner token, so a superseded owner
        can never move its file into the checkpoint."""

        fd, tmp_path = tempfile.mkstemp(
            prefix=f"tmp-{self.owner_id}-",
            suffix=f"-{filename}",
            dir=self.job_dir,
        )
        os.close(fd)
        try:
            write_fn(tmp_path)
            os.replace(tmp_path, os.path.join(self.job_dir, filename))
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _batch_filename(self, batch_index: int, name: str) -> str:
        return f"batch-{batch_index:05d}-{name}.npz"

    def _safe_project_name(self, project_id: str) -> str:
        return "".join(
            ch if (ch.isalnum() or ch in "._-") else "_" for ch in project_id
        )

    def _batch_record(self, batch_index: int) -> dict[str, Any]:
        return self._manifest["batches"].setdefault(str(batch_index), {})

    def is_batch_complete(self, batch_index: int) -> bool:
        """Return True if all project outputs (and, for evaluation jobs,
        the gold subject array) of the batch have been committed."""

        record = self._manifest["batches"].get(str(batch_index))
        if not record:
            return False
        projects = record.get("projects", [])
        if not all(pid in projects for pid in self.fingerprint.project_ids):
            return False
        if self.fingerprint.requires_gold() and not record.get("gold"):
            return False
        return True

    def pending_indices(self) -> list[int]:
        """Return the indices of batches that are not committed yet, in
        original order."""

        return [
            batch_index
            for batch_index in range(self.n_batches)
            if not self.is_batch_complete(batch_index)
        ]

    def load_batch(
        self, batch_index: int
    ) -> tuple[dict[str, SuggestionBatch], scipy.sparse.csr_array | None]:
        """Load the committed suggestions (and gold subjects, if any)
        for a batch."""

        from annif.suggestion import SuggestionBatch

        hit_sets = {}
        for project_id in self.fingerprint.project_ids:
            path = os.path.join(
                self.job_dir,
                self._batch_filename(batch_index, self._safe_project_name(project_id)),
            )
            hit_sets[project_id] = SuggestionBatch(scipy.sparse.load_npz(path))
        gold = None
        if self.fingerprint.requires_gold():
            gold_path = os.path.join(
                self.job_dir, self._batch_filename(batch_index, "gold")
            )
            gold = scipy.sparse.load_npz(gold_path)
        return hit_sets, gold

    def save_batch(
        self,
        batch_index: int,
        hit_sets: dict[str, SuggestionBatch],
        gold: scipy.sparse.csr_array | None,
        owner_id: str,
    ) -> None:
        """Commit the output of a confirmed batch.

        Raises StaleOwnerError if the given owner token does not belong
        to the current owner of the checkpoint; in that case nothing is
        written and the current checkpoint is left untouched."""

        self._check_ownership(owner_id)
        self._manifest = self._read_manifest()

        for project_id, suggestion_batch in hit_sets.items():
            filename = self._batch_filename(
                batch_index, self._safe_project_name(project_id)
            )
            self._atomic_write(
                filename,
                lambda tmp_path, arr=suggestion_batch.array: scipy.sparse.save_npz(
                    tmp_path, arr
                ),
            )

        if gold is not None:
            filename = self._batch_filename(batch_index, "gold")
            self._atomic_write(
                filename,
                lambda tmp_path: scipy.sparse.save_npz(tmp_path, gold),
            )

        record = self._batch_record(batch_index)
        record["projects"] = list(hit_sets.keys())
        if gold is not None:
            record["gold"] = True
        self._manifest["failures"] = [
            failure
            for failure in self._manifest["failures"]
            if failure["batch"] != batch_index
        ]
        self._write_manifest()

    def record_failure(
        self,
        batch_index: int,
        documents: list[list],
        project_id: str | None,
        error: str,
        owner_id: str,
    ) -> None:
        """Record a failed batch together with the identities of its
        documents, so that the batch is retried when the job resumes."""

        self._check_ownership(owner_id)
        self._manifest = self._read_manifest()
        failures = self._manifest["failures"]
        for failure in failures:
            if failure["batch"] == batch_index and failure.get("project") == project_id:
                failure["error"] = error
                failure["documents"] = documents
                break
        else:
            failures.append(
                {
                    "batch": batch_index,
                    "project": project_id,
                    "documents": documents,
                    "error": error,
                }
            )
        self._write_manifest()

    def is_complete(self) -> bool:
        """Return True if every batch is committed and no failure is
        pending."""

        return not self.pending_indices() and not self.failures
