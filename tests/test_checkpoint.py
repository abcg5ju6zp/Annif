"""Unit tests for resumable batch job checkpoints in Annif"""

import json
import os
import os.path
import shutil

import numpy as np
import pytest
from scipy.sparse import csr_array, load_npz

from annif.checkpoint import (
    BatchCheckpoint,
    JobFingerprint,
    StaleOwnerError,
    document_identity,
    format_document_identity,
)
from annif.corpus import Document
from annif.suggestion import SuggestionBatch


def _batch(n_docs, prefix, with_ids=False):
    return [
        Document(
            text=f"text {prefix} {idx}",
            document_id=f"id-{prefix}-{idx}" if with_ids else None,
        )
        for idx in range(n_docs)
    ]


def _fingerprint(
    doc_batches,
    project_ids=("proj-a",),
    job_type="eval",
    limit=10,
    threshold=0.0,
    backend_params=None,
    config_value="v1",
):
    return JobFingerprint(
        job_type=job_type,
        project_ids=list(project_ids),
        projects={
            project_id: {
                "config": [["backend", "dummy"], ["value", config_value]],
                "vocab": {
                    "spec": "dummy",
                    "size": 4,
                    "subjects_hash": f"hash-{config_value}",
                },
                "model_version": f"model-{config_value}",
            }
            for project_id in project_ids
        },
        backend_params=backend_params or {},
        limit=limit,
        threshold=threshold,
        documents=[[document_identity(doc) for doc in batch] for batch in doc_batches],
    )


def _suggestion_matrix(n_docs, n_subjects=4, project_seed=0):
    rows, cols, data = [], [], []
    for doc_idx in range(n_docs):
        rows.append(doc_idx)
        cols.append((doc_idx + project_seed) % n_subjects)
        data.append(0.5 + 0.1 * project_seed)
    return csr_array((data, (rows, cols)), shape=(n_docs, n_subjects), dtype=np.float32)


def _gold_matrix(n_docs, n_subjects=4):
    rows = list(range(n_docs))
    cols = [doc_idx % n_subjects for doc_idx in range(n_docs)]
    return csr_array(
        ([True] * n_docs, (rows, cols)), shape=(n_docs, n_subjects), dtype=bool
    )


def test_document_identity_precedence():
    assert document_identity(Document(text="t", document_id="x")) == ["id", "x"]
    assert document_identity(Document(text="t", file_path="/a/b")) == [
        "path",
        "/a/b",
    ]
    identity = document_identity(Document(text="hello"))
    assert identity[0] == "text"
    assert len(identity[1]) == 64
    # identity is stable for identical text and changes with the text
    assert identity == document_identity(Document(text="hello"))
    assert identity != document_identity(Document(text="hello2"))


def test_format_document_identity():
    assert "document 'x'" == format_document_identity(["id", "x"])
    assert "/a/b" in format_document_identity(["path", "/a/b"])
    assert "hash" in format_document_identity(["text", "abc"])


def test_fingerprint_key_stable_and_sensitive():
    batches = [_batch(2, "a"), _batch(1, "b")]
    fp1 = _fingerprint(batches)
    fp2 = _fingerprint(batches)
    assert fp1.key == fp2.key

    assert fp1.key != _fingerprint(batches, limit=5).key
    assert fp1.key != _fingerprint(batches, threshold=0.5).key
    assert (
        fp1.key != _fingerprint(batches, backend_params={"dummy": {"score": "0"}}).key
    )
    assert fp1.key != _fingerprint(batches, config_value="v2").key
    # document content changed
    changed = [_batch(2, "a"), _batch(1, "DIFFERENT")]
    assert fp1.key != _fingerprint(changed).key
    # document order changed (batches swapped)
    swapped = [batches[1], batches[0]]
    assert fp1.key != _fingerprint(swapped).key


def test_fingerprint_serialization_roundtrip():
    fp = _fingerprint([_batch(2, "a")], backend_params={"dummy": {"score": "1"}})
    restored = JobFingerprint.from_dict(json.loads(json.dumps(fp.to_dict())))
    assert restored == fp
    assert restored.key == fp.key


def test_checkpoint_lifecycle(tmpdir):
    batches = [_batch(2, "a"), _batch(2, "b"), _batch(1, "c")]
    fp = _fingerprint(batches)
    checkpoint = BatchCheckpoint.open(str(tmpdir), fp)
    assert checkpoint.pending_indices() == [0, 1, 2]
    assert not checkpoint.is_complete()

    gold0 = _gold_matrix(2)
    checkpoint.save_batch(
        0,
        {"proj-a": SuggestionBatch(_suggestion_matrix(2))},
        gold0,
        checkpoint.owner_id,
    )
    assert checkpoint.pending_indices() == [1, 2]
    assert not checkpoint.is_complete()

    # reopen: batch 0 is replayed from the checkpoint with a new owner
    resumed = BatchCheckpoint.open(str(tmpdir), fp)
    assert resumed.owner_id != checkpoint.owner_id
    assert resumed.pending_indices() == [1, 2]
    hit_sets, gold = resumed.load_batch(0)
    assert (hit_sets["proj-a"].array != _suggestion_matrix(2)).nnz == 0
    assert (gold != gold0).nnz == 0

    resumed.save_batch(
        1,
        {"proj-a": SuggestionBatch(_suggestion_matrix(2, project_seed=1))},
        _gold_matrix(2),
        resumed.owner_id,
    )
    resumed.save_batch(
        2,
        {"proj-a": SuggestionBatch(_suggestion_matrix(1, project_seed=2))},
        _gold_matrix(1),
        resumed.owner_id,
    )
    assert resumed.pending_indices() == []
    assert resumed.is_complete()


def test_checkpoint_npz_files_are_atomic_and_named_per_batch(tmpdir):
    batches = [_batch(1, "a")]
    fp = _fingerprint(batches)
    checkpoint = BatchCheckpoint.open(str(tmpdir), fp)
    checkpoint.save_batch(
        0,
        {"proj-a": SuggestionBatch(_suggestion_matrix(1))},
        _gold_matrix(1),
        checkpoint.owner_id,
    )
    files = os.listdir(checkpoint.job_dir)
    assert "manifest.json" in files
    assert "batch-00000-proj-a.npz" in files
    assert "batch-00000-gold.npz" in files
    # no leftover temporary files
    assert not [f for f in files if f.startswith("tmp-")]
    # the stored matrix is directly readable
    loaded = load_npz(os.path.join(checkpoint.job_dir, "batch-00000-gold.npz"))
    assert loaded.shape == (1, 4)


def test_stale_owner_cannot_commit_after_takeover(tmpdir):
    batches = [_batch(2, "a"), _batch(2, "b")]
    fp = _fingerprint(batches)
    old_owner = BatchCheckpoint.open(str(tmpdir), fp)
    old_token = old_owner.owner_id

    # a new run takes over the checkpoint
    new_owner = BatchCheckpoint.open(str(tmpdir), fp)
    assert new_owner.owner_id != old_token
    new_owner.save_batch(
        0,
        {"proj-a": SuggestionBatch(_suggestion_matrix(2))},
        _gold_matrix(2),
        new_owner.owner_id,
    )

    # a late result from the superseded owner must not cross the boundary
    with pytest.raises(StaleOwnerError):
        old_owner.save_batch(
            1,
            {"proj-a": SuggestionBatch(_suggestion_matrix(2))},
            _gold_matrix(2),
            old_token,
        )
    with pytest.raises(StaleOwnerError):
        old_owner.record_failure(
            1, [["id", "late"]], "proj-a", "late result", old_token
        )
    # new checkpoint is intact and the stale batch stays pending
    current = BatchCheckpoint.open(str(tmpdir), fp)
    assert current.pending_indices() == [1]
    assert current.failures == []


def test_checkpoint_with_mismatched_fingerprint_is_discarded(tmpdir):
    batches_v1 = [_batch(2, "a")]
    fp_v1 = _fingerprint(batches_v1, config_value="v1")
    checkpoint_v1 = BatchCheckpoint.open(str(tmpdir), fp_v1)
    checkpoint_v1.save_batch(
        0,
        {"proj-a": SuggestionBatch(_suggestion_matrix(2))},
        _gold_matrix(2),
        checkpoint_v1.owner_id,
    )

    # a changed input/model produces an independent job directory, so old
    # and new results can never be combined
    fp_v2 = _fingerprint(batches_v1, config_value="v2")
    checkpoint_v2 = BatchCheckpoint.open(str(tmpdir), fp_v2)
    assert checkpoint_v2.job_dir != checkpoint_v1.job_dir
    assert checkpoint_v2.pending_indices() == [0]
    assert os.path.exists(checkpoint_v1.job_dir)

    # if the expected directory happens to contain a manifest with a
    # mismatching fingerprint, it is wiped rather than reused
    foreign_dir = checkpoint_v2.job_dir
    shutil.rmtree(foreign_dir)
    shutil.move(checkpoint_v1.job_dir, foreign_dir)
    reopened = BatchCheckpoint.open(str(tmpdir), fp_v2)
    assert reopened.job_dir == foreign_dir
    assert reopened.pending_indices() == [0]
    with open(os.path.join(foreign_dir, "manifest.json"), encoding="utf-8") as f:
        assert json.load(f)["fingerprint"] == fp_v2.to_dict()


def test_failure_recorded_and_cleared_on_resume(tmpdir):
    batches = [_batch(2, "a"), _batch(2, "b")]
    fp = _fingerprint(batches)
    checkpoint = BatchCheckpoint.open(str(tmpdir), fp)
    checkpoint.record_failure(
        1,
        fp.documents[1],
        "proj-a",
        "RuntimeError: boom",
        checkpoint.owner_id,
    )
    # failed batch is not complete and must be retried
    assert checkpoint.pending_indices() == [0, 1]
    assert len(checkpoint.failures) == 1

    resumed = BatchCheckpoint.open(str(tmpdir), fp)
    assert resumed.failures[0]["error"] == "RuntimeError: boom"
    # other batches are still usable while the failed one is retried
    resumed.save_batch(
        0,
        {"proj-a": SuggestionBatch(_suggestion_matrix(2))},
        _gold_matrix(2),
        resumed.owner_id,
    )
    assert resumed.pending_indices() == [1]
    # retrying the batch successfully clears its failure record
    resumed.save_batch(
        1,
        {"proj-a": SuggestionBatch(_suggestion_matrix(2))},
        _gold_matrix(2),
        resumed.owner_id,
    )
    assert resumed.failures == []
    assert resumed.is_complete()


def test_multi_project_batch_requires_all_projects(tmpdir):
    batches = [_batch(2, "a")]
    fp = _fingerprint(batches, project_ids=("proj-a", "proj-b"))
    checkpoint = BatchCheckpoint.open(str(tmpdir), fp)
    checkpoint.save_batch(
        0,
        {"proj-a": SuggestionBatch(_suggestion_matrix(2))},
        _gold_matrix(2),
        checkpoint.owner_id,
    )
    # one project failing leaves the batch pending but must not lose the
    # suggestions of the project that succeeded
    assert checkpoint.pending_indices() == [0]
    assert os.path.exists(os.path.join(checkpoint.job_dir, "batch-00000-proj-a.npz"))
    assert not os.path.exists(
        os.path.join(checkpoint.job_dir, "batch-00000-proj-b.npz")
    )

    checkpoint.save_batch(
        0,
        {
            "proj-a": SuggestionBatch(_suggestion_matrix(2)),
            "proj-b": SuggestionBatch(_suggestion_matrix(2, project_seed=1)),
        },
        _gold_matrix(2),
        checkpoint.owner_id,
    )
    assert checkpoint.is_complete()
    hit_sets, gold = checkpoint.load_batch(0)
    assert set(hit_sets.keys()) == {"proj-a", "proj-b"}
    assert (hit_sets["proj-b"].array != _suggestion_matrix(2, project_seed=1)).nnz == 0
    assert gold is not None


def test_suggest_job_does_not_require_gold(tmpdir):
    batches = [_batch(2, "a")]
    fp = _fingerprint(batches, job_type="suggest")
    checkpoint = BatchCheckpoint.open(str(tmpdir), fp)
    assert not checkpoint.is_batch_complete(0)
    checkpoint.save_batch(
        0,
        {"proj-a": SuggestionBatch(_suggestion_matrix(2))},
        None,
        checkpoint.owner_id,
    )
    assert checkpoint.is_complete()
    hit_sets, gold = checkpoint.load_batch(0)
    assert gold is None
    assert "proj-a" in hit_sets
