"""Tests for atomic model generation management."""

import json
import os
import os.path
import threading

import pytest

from annif.backend.tfidf import TFIDFBackend
from annif.corpus import Document, DocumentList
from annif.exception import NotSupportedException, OperationFailedException
from annif.model import (
    GENERATIONS_DIRNAME,
    KEEP_GENERATIONS,
    MANIFEST_FILENAME,
    POINTER_FILENAME,
    STAGING_DIRNAME,
    ModelGenerationManager,
    corpus_fingerprint,
)

# -- helpers ----------------------------------------------------------------


class FakeProject:
    """Minimal project stand-in for manager-level unit tests."""

    def __init__(self, datadir, project_id="fake"):
        self._datadir_path = str(datadir)
        self.project_id = project_id

    @property
    def datadir(self):
        os.makedirs(self._datadir_path, exist_ok=True)
        return self._datadir_path


def _pointer(manager):
    return manager.read_pointer()


def _generation_ids(manager):
    if not os.path.isdir(manager.generations_dir):
        return set()
    return {
        entry
        for entry in os.listdir(manager.generations_dir)
        if os.path.isdir(manager.generation_dir(entry))
    }


def _reset_backend_cache(project):
    """Simulate a process restart for the active backend cache."""
    project._backend = None


# -- untrained / legacy resolution -----------------------------------------


def test_untrained_project_uses_datadir_root(registry):
    project = registry.get_project("tfidf-fi")
    manager = project.model_manager
    assert manager.read_pointer() is None
    assert project.active_model_dir() == project._datadir_path
    assert not project.is_trained


def test_legacy_root_files_are_served_without_pointer(registry, testdatadir):
    # a project directory with model files at the root but no generation
    # pointer predates model generations and keeps working unchanged
    project = registry.get_project("tfidf-fi")
    project_dir = testdatadir.join("projects/tfidf-fi")
    project_dir.ensure("tfidf-matrix.npz").write("legacy")
    try:
        assert project.active_model_dir() == project._datadir_path
        assert project.is_trained
    finally:
        project_dir.join("tfidf-matrix.npz").remove()


# -- first training and generation contents --------------------------------


def test_train_publishes_generation(registry, document_corpus, testdatadir):
    project = registry.get_project("tfidf-fi")
    manager = project.model_manager
    assert manager.read_pointer() is None

    project.train(document_corpus)

    pointer = manager.read_pointer()
    assert pointer is not None
    assert pointer["previous"] is None
    generation_id = pointer["generation"]

    gen_dir = manager.generation_dir(generation_id)
    assert os.path.isdir(gen_dir)
    manifest = manager.read_manifest(generation_id)
    assert manifest is not None
    assert manifest["operation"] == "train"
    assert manifest["parent"] is None

    # the manifest pins the vocabulary, corpus, analyzer and backend params
    fingerprint = manifest["fingerprint"]
    assert fingerprint["corpus"]["documents"] > 0
    assert len(fingerprint["corpus"]["sha256"]) == 64
    assert fingerprint["analyzer"]["spec"]
    assert fingerprint["vocab"]["id"] == "yso"
    assert fingerprint["backend"]["id"] == "tfidf"
    assert "limit" in fingerprint["backend"]["params"]

    # all model files are recorded with size and checksum; train/tmp files
    # and the bookkeeping files are not part of the model
    paths = {entry["path"] for entry in manifest["files"]}
    assert "tfidf-matrix.npz" in paths
    assert "vectorizer" in paths
    assert all(entry["size"] > 0 for entry in manifest["files"])
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert not any("train" in path for path in paths)

    # no staging directory or lock artifacts remain after publishing
    assert not os.path.isdir(manager.staging_root)
    assert project.is_trained
    # the active backend instance is bound to the generation directory
    assert project.backend.datadir == gen_dir


def test_suggest_uses_complete_generation_after_train(registry, document_corpus):
    project = registry.get_project("tfidf-fi")
    project.train(document_corpus)
    generation_id = project.model_manager.active_generation_id()
    doc = next(document_corpus.documents)
    project.suggest([Document(text=doc.text)])
    assert project.backend.datadir.endswith(
        os.path.join(GENERATIONS_DIRNAME, generation_id)
    )


# -- retraining, pruning and pointer chain ---------------------------------


def test_retrain_publishes_new_generation_and_prunes(registry, document_corpus):
    project = registry.get_project("tfidf-fi")
    manager = project.model_manager

    project.train(document_corpus)
    first = manager.active_generation_id()

    project.train(document_corpus)
    second = manager.active_generation_id()
    assert second != first
    pointer = manager.read_pointer()
    assert pointer["generation"] == second
    assert pointer["previous"] == first
    # the previous generation is retained for immediate rollback
    assert first in _generation_ids(manager)

    project.train(document_corpus)
    third = manager.active_generation_id()
    successful = {
        gen_id
        for gen_id in _generation_ids(manager)
        if os.path.exists(
            os.path.join(manager.generation_dir(gen_id), MANIFEST_FILENAME)
        )
        and not os.path.exists(
            os.path.join(manager.generation_dir(gen_id), "activation-error.json")
        )
    }
    assert third in successful
    assert second in successful
    assert len(successful) == KEEP_GENERATIONS
    assert first not in successful
    assert manager.read_pointer()["previous"] == second


def test_modification_time_reflects_active_generation(registry, document_corpus):
    from datetime import datetime, timedelta, timezone

    project = registry.get_project("tfidf-fi")
    project.train(document_corpus)
    assert datetime.now(timezone.utc) - project.modification_time < timedelta(1)


# -- training failure keeps the old model ----------------------------------


def test_failed_training_cleans_candidate_and_keeps_old_model(
    registry, document_corpus, empty_corpus
):
    project = registry.get_project("tfidf-fi")
    project.train(document_corpus)
    manager = project.model_manager
    active_before = manager.active_generation_id()
    backend_before = project.backend
    gen_dirs_before = _generation_ids(manager)

    with pytest.raises(NotSupportedException):
        project.train(empty_corpus)

    # no candidate leaked and the old generation is still active and usable
    assert not os.path.isdir(manager.staging_root)
    assert manager.active_generation_id() == active_before
    assert _generation_ids(manager) == gen_dirs_before
    assert project.backend is backend_before
    assert project.is_trained


# -- activation failure rolls the published generation back ----------------


def test_activation_failure_rolls_back_and_keeps_diagnostics(
    registry, document_corpus, monkeypatch
):
    project = registry.get_project("tfidf-fi")
    project.train(document_corpus)
    manager = project.model_manager
    previous = manager.active_generation_id()
    backend_before = project.backend

    real_initialize = TFIDFBackend.initialize

    def initialize_failing_after_publish(self, parallel=False):
        real_initialize(self, parallel)
        # the probe loads from the staging directory and must succeed; only
        # activation from the published generation directory fails
        if STAGING_DIRNAME not in self.datadir:
            raise OperationFailedException("simulated activation failure")

    monkeypatch.setattr(TFIDFBackend, "initialize", initialize_failing_after_publish)

    with pytest.raises(OperationFailedException, match="simulated activation"):
        project.train(document_corpus)

    # the active generation is rolled back and the old instance keeps serving
    pointer = manager.read_pointer()
    assert pointer["generation"] == previous
    assert project.backend is backend_before
    assert project.is_trained

    # the failed generation and diagnostics are retained
    failed_id = pointer["rolled_back_from"]
    failed_dir = manager.generation_dir(failed_id)
    assert os.path.isdir(failed_dir)
    assert os.path.exists(os.path.join(failed_dir, "activation-error.json"))
    diagnostics = os.path.join(manager.diagnostics_dir, f"{failed_id}.json")
    assert os.path.exists(diagnostics)
    with open(diagnostics, encoding="utf-8") as diag_file:
        assert "simulated activation failure" in json.load(diag_file)["error"]


# -- restart completes the switch from the manifest ------------------------


def test_restart_loads_active_generation_from_pointer(
    registry, document_corpus, testdatadir
):
    project = registry.get_project("tfidf-fi")
    project.train(document_corpus)
    generation_id = project.model_manager.active_generation_id()

    # drop all in-memory caches, as a restart would
    _reset_backend_cache(project)
    assert project.active_model_dir() == project.model_manager.generation_dir(
        generation_id
    )
    assert project.is_trained
    doc = next(document_corpus.documents)
    project.suggest([Document(text=doc.text)])


def test_incomplete_active_generation_falls_back_on_restart(registry, document_corpus):
    project = registry.get_project("tfidf-fi")
    project.train(document_corpus)
    first = project.model_manager.active_generation_id()
    project.train(document_corpus)
    second = project.model_manager.active_generation_id()

    # corrupt the newest generation after publish
    os.remove(
        os.path.join(project.model_manager.generation_dir(second), "tfidf-matrix.npz")
    )
    _reset_backend_cache(project)

    # resolution falls back to the complete previous generation
    assert project.active_model_dir() == project.model_manager.generation_dir(first)
    assert project.is_trained


# -- online requests while retraining --------------------------------------


def test_suggest_keeps_serving_old_generation_during_retrain(
    registry, document_corpus, monkeypatch
):
    project = registry.get_project("tfidf-fi")
    manager = project.model_manager
    project.train(document_corpus)
    first_id = manager.active_generation_id()

    entered_training = threading.Event()
    release_training = threading.Event()
    real_train = TFIDFBackend._train

    def blocking_train(self, corpus, params, jobs=0):
        entered_training.set()
        assert release_training.wait(timeout=10)
        return real_train(self, corpus, params, jobs)

    monkeypatch.setattr(TFIDFBackend, "_train", blocking_train)

    errors = []

    def retrain():
        try:
            project.train(document_corpus)
        except Exception as err:  # noqa: BLE001
            errors.append(err)

    trainer = threading.Thread(target=retrain)
    trainer.start()
    assert entered_training.wait(timeout=10)
    try:
        # the pointer and the cached active instance still refer to the old
        # generation, and online suggest requests complete against it
        assert manager.active_generation_id() == first_id
        assert project.is_trained
        assert project.backend.datadir.endswith(first_id)
        doc = next(document_corpus.documents)
        project.suggest([Document(text=doc.text)])
    finally:
        release_training.set()
    trainer.join(timeout=30)
    assert not errors
    # only after the switch completes do new requests see the new generation
    second_id = manager.active_generation_id()
    assert second_id != first_id
    assert project.backend.datadir.endswith(second_id)


# -- unique ownership -------------------------------------------------------


def test_concurrent_train_is_rejected(registry, document_corpus):
    project = registry.get_project("tfidf-fi")
    manager = project.model_manager

    with manager.ownership("train"):
        # a second owner in the same process cannot enter and must not train
        with pytest.raises(OperationFailedException, match="already in progress"):
            with manager.ownership("train"):
                pass  # pragma: no cover

    # after the owner exits, the next operation succeeds
    project.train(document_corpus)


def test_late_products_are_not_committed_after_ownership_lost(tmpdir, monkeypatch):
    manager = ModelGenerationManager(FakeProject(tmpdir.mkdir("proj")))
    with manager.ownership("train") as owner:
        # a model file appears in the candidate directory
        with open(os.path.join(owner.staging_dir, "model.bin"), "wb") as model:
            model.write(b"candidate")
        # ownership is stolen by another owner (e.g. stale lock replaced)
        with open(manager.lock_path, "w", encoding="utf-8") as lock_file:
            json.dump({"token": "someone-else"}, lock_file)
        with pytest.raises(OperationFailedException, match="lost ownership"):
            manager.commit(owner, "late-generation")
        # the late products never reach the generations directory
        assert not os.path.exists(manager.generations_dir)


def test_stale_staging_is_cleaned_for_new_owner(tmpdir):
    manager = ModelGenerationManager(FakeProject(tmpdir.mkdir("proj")))
    with manager.ownership("train") as owner:
        stale = owner.staging_dir
        assert os.path.isdir(stale)
    # simulate a crashed owner leaving the staging directory behind
    os.makedirs(stale, exist_ok=True)
    with open(os.path.join(stale, "leftover"), "w") as leftover:
        leftover.write("partial")
    with manager.ownership("train"):
        assert not os.path.exists(stale)


def test_seed_from_active_copies_model_files(registry, document_corpus, tmpdir):
    project = registry.get_project("tfidf-fi")
    project.train(document_corpus)
    manager = project.model_manager
    staging = str(tmpdir.mkdir("candidate"))

    manager.seed_from_active(staging, include_train_data=False)
    assert os.path.exists(os.path.join(staging, "tfidf-matrix.npz"))
    assert os.path.exists(os.path.join(staging, "vectorizer"))
    assert not os.path.exists(os.path.join(staging, MANIFEST_FILENAME))


# -- corpus fingerprint -----------------------------------------------------


def test_corpus_fingerprint_is_stable_and_sensitive():
    docs_a = [
        Document(text="hello world", subject_set=None),
        Document(text="second document", subject_set=None),
    ]
    docs_a_copy = [
        Document(text="hello world", subject_set=None),
        Document(text="second document", subject_set=None),
    ]
    docs_b = [Document(text="different content", subject_set=None)]
    fp_a = corpus_fingerprint(DocumentList(docs_a))
    fp_a_copy = corpus_fingerprint(DocumentList(docs_a_copy))
    fp_b = corpus_fingerprint(DocumentList(docs_b))
    assert fp_a == fp_a_copy
    assert fp_a != fp_b
    assert fp_a["documents"] == 2


# -- backends without model files are unaffected ---------------------------


def test_nonversioned_backend_uses_datadir_root(registry):
    project = registry.get_project("dummy-fi")
    assert project.backend.versioned_model is False
    assert project.backend.datadir == project._datadir_path
    project.learn(
        DocumentList(
            [
                Document(
                    text="learning text",
                    subject_set=None,
                    metadata={},
                )
            ]
        )
    )
    assert not os.path.isdir(os.path.join(project._datadir_path, GENERATIONS_DIRNAME))
    assert not os.path.exists(os.path.join(project._datadir_path, POINTER_FILENAME))


def test_ensemble_backend_is_nonversioned(registry):
    project = registry.get_project("ensemble")
    assert project.backend.versioned_model is False
    assert project.backend.datadir == project._datadir_path
