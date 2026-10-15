"""Atomic model generation management for Annif projects.

A project that is serving suggestion requests online can be retrained by its
maintainer.  Retraining typically produces several model files, and writing
them one by one directly into the live data directory would leave a window
where an online request could load a mixture of old and new files while
``is_trained`` still reports the project as usable.

This module implements safe model replacements using *model generations*:

* every train/learn operation is performed by a single owner (an exclusive
  lock shared between threads and processes) in an isolated staging
  directory;
* the vocabulary, corpus, analyzer and backend parameter versions are
  recorded in a fingerprint, so every generation is fully self-describing;
* after training, the files are validated (no partial temporary files, all
  files present and matching their recorded checksums) and the candidate is
  exercised with a controlled suggestion probe that loads the model from
  scratch;
* the generation and its manifest are published atomically (directory rename
  followed by an atomic replacement of the current-generation pointer);
* should the freshly published files fail to initialize as a live backend
  instance, the pointer is rolled back to the previous generation and
  diagnostics are retained;
* a process restart completes the in-memory switch by reading the pointer
  and validating the generation against its manifest.

Backends that do not persist trained model files (e.g. the dummy, HTTP,
yake and ensemble backends) bypass generation management entirely and keep
using the project data directory root exactly as before.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import os.path
import shutil
import socket
import threading
import uuid
from datetime import datetime, timezone
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, NamedTuple

from annif import logger
from annif.corpus import Document
from annif.exception import OperationFailedException

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annif.backend import AnnifBackend
    from annif.corpus.document import DocumentCorpus

#: directory that holds all published model generations
GENERATIONS_DIRNAME = "generations"

#: directory that holds in-progress, not yet published candidate generations
STAGING_DIRNAME = ".staging"

#: name of the lock file used to establish the unique training owner
LOCK_FILENAME = ".model-lock"

#: name of the atomically replaced current-generation pointer file
POINTER_FILENAME = "current.json"

#: name of the manifest stored inside every generation directory
MANIFEST_FILENAME = "manifest.json"

#: marker left inside a generation whose live initialization failed
ACTIVATION_ERROR_FILENAME = "activation-error.json"

#: directory (under the project datadir) for activation diagnostics
DIAGNOSTICS_DIRNAME = "diagnostics"

#: glob patterns identifying cached training data or partial temporary files
TRAIN_FILE_PATTERNS = ("*-train*", "tmp-*")

#: number of successfully published generations to retain per project
KEEP_GENERATIONS = 2

#: text used for the controlled post-training suggestion probe
PROBE_TEXT = "annif model activation probe"

MANIFEST_SCHEMA_VERSION = 1

# in-process locks, keyed by the project data directory path
_process_locks: dict[str, threading.Lock] = {}
_process_locks_guard = threading.Lock()


class ModelOwner(NamedTuple):
    """Handle for the unique owner of a model update operation."""

    token: str
    operation: str
    staging_dir: str


def _is_ignored_path(name: str) -> bool:
    """Return True if the given file or directory name is cached training
    data or a temporary file and thus never part of a published model."""

    return any(fnmatch(name, pattern) for pattern in TRAIN_FILE_PATTERNS)


def _iter_files(base_dir: str, *, include_train_data: bool = False) -> Iterator[str]:
    """Iterate over the regular files below base_dir.  Bookkeeping files and
    (unless include_train_data is True) cached training data are skipped."""

    for dirpath, dirnames, filenames in os.walk(base_dir):
        if not include_train_data:
            dirnames[:] = [
                name
                for name in dirnames
                if not _is_ignored_path(name)
                and name not in (MANIFEST_FILENAME, ACTIVATION_ERROR_FILENAME)
            ]
        for filename in sorted(filenames):
            if filename in (MANIFEST_FILENAME, ACTIVATION_ERROR_FILENAME):
                continue
            if not include_train_data and _is_ignored_path(filename):
                continue
            yield os.path.join(dirpath, filename)


def _sha256_file(path: str) -> tuple[int, str]:
    """Return the size and SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as model_file:
        while chunk := model_file.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def corpus_fingerprint(corpus: DocumentCorpus) -> dict[str, Any]:
    """Calculate a fingerprint of the (already transformed) training
    corpus."""

    digest = hashlib.sha256()
    n_documents = 0
    for doc in corpus.documents:
        digest.update(doc.text.encode("utf-8", "replace"))
        subject_ids = sorted(doc.subject_set) if doc.subject_set is not None else []
        digest.update(b"\0")
        digest.update(",".join(str(sid) for sid in subject_ids).encode("ascii"))
        digest.update(b"\n")
        n_documents += 1
    return {"documents": n_documents, "sha256": digest.hexdigest()}


def probe_backend(backend: AnnifBackend, params: dict[str, Any]) -> None:
    """Probe a candidate backend by initializing it from its model files and
    performing a controlled suggestion call.  Any failure is allowed to
    propagate so that the candidate is rejected."""

    backend.initialize()
    backend.suggest([Document(text=PROBE_TEXT)], params)


class ModelGenerationManager:
    """Manage the on-disk model generations of a single project."""

    def __init__(self, project) -> None:
        self.project = project
        self.root = project._datadir_path

    # -- paths ------------------------------------------------------------

    @property
    def generations_dir(self) -> str:
        return os.path.join(self.root, GENERATIONS_DIRNAME)

    @property
    def staging_root(self) -> str:
        return os.path.join(self.root, STAGING_DIRNAME)

    @property
    def lock_path(self) -> str:
        return os.path.join(self.root, LOCK_FILENAME)

    @property
    def pointer_path(self) -> str:
        return os.path.join(self.root, POINTER_FILENAME)

    @property
    def diagnostics_dir(self) -> str:
        return os.path.join(self.root, DIAGNOSTICS_DIRNAME)

    def generation_dir(self, generation_id: str) -> str:
        return os.path.join(self.generations_dir, generation_id)

    # -- ownership --------------------------------------------------------

    @contextlib.contextmanager
    def ownership(self, operation: str) -> Iterator[ModelOwner]:
        """Acquire exclusive ownership for a model update operation.

        Only one thread or process may hold ownership for the project at any
        time; attempts to acquire it while another owner is active fail
        immediately.  The lock is released automatically when the process
        exits, so crashed owners never block future operations."""

        os.makedirs(self.root, exist_ok=True)

        with _process_locks_guard:
            thread_lock = _process_locks.setdefault(self.root, threading.Lock())
        if not thread_lock.acquire(blocking=False):
            raise OperationFailedException(
                f"{operation} is already in progress for project "
                f"'{self.project.project_id}'",
                project_id=self.project.project_id,
            )

        lock_file = None
        staging_dir: str | None = None
        try:
            lock_file = open(self.lock_path, "a+")
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise OperationFailedException(
                    f"{operation} is already in progress for project "
                    f"'{self.project.project_id}' in another process",
                    project_id=self.project.project_id,
                )

            token = uuid.uuid4().hex
            owner_info = {
                "token": token,
                "operation": operation,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            lock_file.seek(0)
            lock_file.truncate()
            json.dump(owner_info, lock_file)
            lock_file.flush()
            os.fsync(lock_file.fileno())

            self._cleanup_staging()
            staging_dir = os.path.join(self.staging_root, token)
            os.makedirs(staging_dir)
            logger.debug(
                "Project '%s': %s owner token %s using staging dir %s",
                self.project.project_id,
                operation,
                token,
                staging_dir,
            )
            yield ModelOwner(token=token, operation=operation, staging_dir=staging_dir)
        finally:
            if staging_dir is not None and os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)
            # remove the staging root if it is now empty, so that no
            # bookkeeping directories are left behind
            with contextlib.suppress(OSError):
                if os.path.isdir(self.staging_root) and not os.listdir(
                    self.staging_root
                ):
                    os.rmdir(self.staging_root)
            if lock_file is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
            thread_lock.release()

    def _cleanup_staging(self) -> None:
        """Remove candidate directories abandoned by earlier operations."""

        if not os.path.isdir(self.staging_root):
            return
        for entry in os.listdir(self.staging_root):
            path = os.path.join(self.staging_root, entry)
            if os.path.isdir(path):
                logger.debug(
                    "Project '%s': removing stale candidate directory %s",
                    self.project.project_id,
                    path,
                )
                shutil.rmtree(path, ignore_errors=True)

    def assert_owner(self, owner: ModelOwner) -> None:
        """Verify that the given owner still holds ownership.  This guards
        against committing products produced after ownership was lost."""

        try:
            with open(self.lock_path, encoding="utf-8") as lock_file:
                current = json.load(lock_file)
        except (OSError, ValueError):
            raise OperationFailedException(
                "model ownership cannot be verified",
                project_id=self.project.project_id,
            )
        if current.get("token") != owner.token:
            raise OperationFailedException(
                f"lost ownership of project '{self.project.project_id}'; "
                "discarding the candidate model",
                project_id=self.project.project_id,
            )

    # -- staging / candidate handling ------------------------------------

    def seed_from_active(self, staging_dir: str, *, include_train_data: bool) -> None:
        """Copy the files of the currently active generation into a
        candidate directory (used for learn and cached training)."""

        source = self.active_model_dir()
        if (
            not source
            or not os.path.isdir(source)
            or os.path.abspath(source) == os.path.abspath(staging_dir)
        ):
            return

        def ignore(_directory: str, names: list[str]) -> set[str]:
            ignored = set()
            for name in names:
                # manifests, markers and temporary files are never copied;
                # cached training data is copied only when explicitly asked
                if (
                    name in (MANIFEST_FILENAME, ACTIVATION_ERROR_FILENAME)
                    or fnmatch(name, "tmp-*")
                    or (not include_train_data and fnmatch(name, "*-train*"))
                ):
                    ignored.add(name)
            return ignored

        shutil.copytree(source, staging_dir, dirs_exist_ok=True, ignore=ignore)

    def finalize_candidate(self, staging_dir: str) -> list[dict[str, Any]]:
        """Validate the files in a candidate directory and return their
        manifest entries."""

        leftovers = [
            os.path.join(dirpath, name)
            for dirpath, _dirnames, filenames in os.walk(staging_dir)
            for name in filenames
            if fnmatch(name, "tmp-*")
        ]
        if leftovers:
            raise OperationFailedException(
                "training left incomplete temporary model files: "
                + ", ".join(sorted(leftovers)[:5]),
                project_id=self.project.project_id,
            )

        entries = []
        for path in _iter_files(staging_dir):
            size, checksum = _sha256_file(path)
            if size == 0:
                raise OperationFailedException(
                    f"model file {path} is empty",
                    project_id=self.project.project_id,
                )
            entries.append(
                {
                    "path": os.path.relpath(path, staging_dir),
                    "size": size,
                    "sha256": checksum,
                }
            )
        if not entries:
            raise OperationFailedException(
                "training produced no model files",
                project_id=self.project.project_id,
            )
        return entries

    def write_manifest(
        self,
        staging_dir: str,
        *,
        generation_id: str,
        operation: str,
        parent: str | None,
        fingerprint: dict[str, Any],
        files: list[dict[str, Any]],
    ) -> str:
        """Write the generation manifest into the candidate directory."""

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generation": generation_id,
            "operation": operation,
            "parent": parent,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "annif_version": importlib.metadata.version("annif"),
            "fingerprint": fingerprint,
            "files": sorted(files, key=lambda entry: entry["path"]),
        }
        manifest_path = os.path.join(staging_dir, MANIFEST_FILENAME)
        self._atomic_write_json(manifest_path, manifest)
        return manifest_path

    # -- publish / rollback ----------------------------------------------

    def commit(self, owner: ModelOwner, generation_id: str) -> str:
        """Publish a validated candidate by atomically moving it into the
        generations directory.  Return the published generation path."""

        self.assert_owner(owner)
        target = self.generation_dir(generation_id)
        os.makedirs(self.generations_dir, exist_ok=True)
        if os.path.exists(target):
            raise OperationFailedException(
                f"model generation {generation_id} already exists",
                project_id=self.project.project_id,
            )
        os.rename(owner.staging_dir, target)
        self._fsync_dir(self.generations_dir)
        return target

    def publish_pointer(self, generation_id: str, previous: str | None) -> None:
        """Atomically make the given generation the active generation."""

        os.makedirs(self.root, exist_ok=True)
        pointer = {
            "generation": generation_id,
            "previous": previous,
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_write_json(self.pointer_path, pointer)
        self._fsync_dir(self.root)

    def rollback_after_activation_failure(
        self, generation_id: str, previous: str | None, error: Exception
    ) -> None:
        """Roll the active pointer back to the previous generation (or remove
        it when there is no previous generation) and retain diagnostics."""

        logger.warning(
            "Project '%s': published generation %s failed to initialize "
            "(%s); rolling back to %s",
            self.project.project_id,
            generation_id,
            error,
            previous or "the legacy model",
        )
        if previous:
            self._atomic_write_json(
                self.pointer_path,
                {
                    "generation": previous,
                    "previous": None,
                    "rolled_back_from": generation_id,
                    "rolled_back_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        elif os.path.exists(self.pointer_path):
            os.remove(self.pointer_path)

        diagnostic = {
            "generation": generation_id,
            "stage": "initialize",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        failed_dir = self.generation_dir(generation_id)
        with contextlib.suppress(OSError):
            self._atomic_write_json(
                os.path.join(failed_dir, ACTIVATION_ERROR_FILENAME), diagnostic
            )
        os.makedirs(self.diagnostics_dir, exist_ok=True)
        with contextlib.suppress(OSError):
            self._atomic_write_json(
                os.path.join(self.diagnostics_dir, f"{generation_id}.json"),
                diagnostic,
            )

    def prune_generations(self) -> None:
        """Remove old successful generations, retaining the most recent ones
        and never removing the active, previous or failed generations."""

        if not os.path.isdir(self.generations_dir):
            return
        pointer = self.read_pointer() or {}
        retained = {pointer.get("generation"), pointer.get("previous")}
        retained.discard(None)
        candidates = []
        for entry in os.listdir(self.generations_dir):
            path = self.generation_dir(entry)
            if not os.path.isdir(path) or entry in retained:
                continue
            if os.path.exists(os.path.join(path, ACTIVATION_ERROR_FILENAME)):
                continue  # keep failed generations for diagnosis
            if not os.path.exists(os.path.join(path, MANIFEST_FILENAME)):
                continue  # not a complete published generation
            candidates.append((os.path.getmtime(path), entry, path))
        candidates.sort(reverse=True)
        # the retained generations already count towards the kept total, so
        # only the newest remaining generations are kept on top of them
        keep_extra = max(0, KEEP_GENERATIONS - len(retained))
        for _mtime, _entry, path in candidates[keep_extra:]:
            logger.debug(
                "Project '%s': pruning old model generation %s",
                self.project.project_id,
                path,
            )
            shutil.rmtree(path, ignore_errors=True)

    # -- reading the active generation ------------------------------------

    def read_pointer(self) -> dict[str, Any] | None:
        """Read the current-generation pointer, if any."""

        try:
            with open(self.pointer_path, encoding="utf-8") as pointer_file:
                return json.load(pointer_file)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as err:
            logger.warning(
                "Project '%s': cannot read model generation pointer: %s",
                self.project.project_id,
                err,
            )
            return None

    def read_manifest(self, generation_id: str) -> dict[str, Any] | None:
        """Read the manifest of a generation, if available and valid."""

        try:
            with open(
                os.path.join(self.generation_dir(generation_id), MANIFEST_FILENAME),
                encoding="utf-8",
            ) as manifest_file:
                return json.load(manifest_file)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as err:
            logger.warning(
                "Project '%s': cannot read manifest of generation %s: %s",
                self.project.project_id,
                generation_id,
                err,
            )
            return None

    def active_manifest(self) -> dict[str, Any] | None:
        """Return the manifest of the generation referred to by the pointer."""

        pointer = self.read_pointer()
        if not pointer:
            return None
        return self.read_manifest(pointer["generation"])

    def active_generation_id(self) -> str | None:
        pointer = self.read_pointer()
        return pointer["generation"] if pointer else None

    def generation_is_complete(self, generation_id: str) -> bool:
        """Check that every file recorded in the generation manifest exists
        with the recorded size."""

        gen_dir = self.generation_dir(generation_id)
        manifest = self.read_manifest(generation_id)
        if manifest is None:
            return False
        for entry in manifest.get("files", []):
            path = os.path.join(gen_dir, entry["path"])
            if not os.path.isfile(path):
                return False
            if entry.get("size") is not None and os.path.getsize(path) != entry["size"]:
                return False
        return True

    def active_model_dir(self) -> str:
        """Return the directory holding the active, complete model.

        Falls back to the previous generation referenced by the pointer and
        finally to the project data directory root, which is used for legacy
        models and for projects that have never been trained."""

        pointer = self.read_pointer()
        if pointer:
            generation_id = pointer.get("generation")
            if generation_id and self.generation_is_complete(generation_id):
                return self.generation_dir(generation_id)
            logger.warning(
                "Project '%s': active model generation %s is incomplete; "
                "falling back",
                self.project.project_id,
                generation_id,
            )
            previous = pointer.get("previous")
            if previous and self.generation_is_complete(previous):
                return self.generation_dir(previous)
        return self.project.datadir

    # -- helpers ----------------------------------------------------------

    def new_generation_id(self, fingerprint_digest: str, token: str) -> str:
        """Build a unique, sortable generation identifier."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        return f"{timestamp}-{fingerprint_digest[:8]}-{token[:8]}"

    @staticmethod
    def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        temp_path = f"{path}.tmp-{uuid.uuid4().hex}"
        with open(temp_path, "w", encoding="utf-8") as outfile:
            json.dump(payload, outfile, indent=2, sort_keys=True)
            outfile.flush()
            os.fsync(outfile.fileno())
        os.replace(temp_path, path)

    @staticmethod
    def _fsync_dir(path: str) -> None:
        with contextlib.suppress(OSError):
            dir_fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
