"""Unit tests for parallel processing functionality in Annif"""

import multiprocessing
import multiprocessing.dummy
import multiprocessing.pool

import numpy as np
from scipy.sparse import csr_array

import annif.checkpoint
import annif.parallel
from annif.checkpoint import BatchCheckpoint, JobFingerprint, document_identity
from annif.corpus import Document, SubjectSet
from annif.suggestion import SuggestionBatch


def test_get_pool_0():
    n_jobs, pool_class = annif.parallel.get_pool(0)
    assert n_jobs is None
    assert isinstance(pool_class(), multiprocessing.pool.Pool)


def test_get_pool_1():
    n_jobs, pool_class = annif.parallel.get_pool(1)
    assert n_jobs == 1
    assert pool_class is multiprocessing.dummy.Pool


def test_get_pool_2():
    n_jobs, pool_class = annif.parallel.get_pool(2)
    assert n_jobs == 2
    assert isinstance(pool_class(), multiprocessing.pool.Pool)


class FakeProjectSuggestMap:
    """Drop-in replacement for ProjectSuggestMap in runner tests,
    recording the order and identity of every inference call."""

    def __init__(self, project_ids=("p1", "p2"), fail_projects=(), fail_texts=()):
        self.project_ids = list(project_ids)
        self.fail_projects = set(fail_projects)
        self.fail_texts = set(fail_texts)
        self.calls = []

    def suggest_batch_isolated(self, batch):
        hit_sets, failures = {}, {}
        subject_sets = [doc.subject_set for doc in batch]
        for project_id in self.project_ids:
            for doc in batch:
                self.calls.append((project_id, doc.text))
            if project_id in self.fail_projects or any(
                doc.text in self.fail_texts for doc in batch
            ):
                failures[project_id] = "RuntimeError: simulated worker failure"
            else:
                # encode batch position and project in the score so that
                # suggestion/doc alignment can be verified
                score = (
                    float(batch[0].text[1:])
                    + 1.0
                    + (0.1 if project_id == "p2" else 0.0)
                )
                hit_sets[project_id] = SuggestionBatch(
                    csr_array(
                        ([score] * len(batch), ([0] * len(batch), [0] * len(batch))),
                        shape=(len(batch), 2),
                        dtype=np.float32,
                    )
                )
        return hit_sets, subject_sets, failures


class ExplodingSuggestMap(FakeProjectSuggestMap):
    def suggest_batch_isolated(self, batch):
        self.calls.append(("*", batch[0].text))
        raise RuntimeError("catastrophic failure")


def _make_batches():
    return [
        [Document(text="d0", subject_set=SubjectSet([0]))],
        [Document(text="d1", subject_set=SubjectSet([1]))],
        [Document(text="d2", subject_set=SubjectSet([0]))],
    ]


def _make_fingerprint(doc_batches, project_ids=("p1", "p2")):
    return JobFingerprint(
        job_type="eval",
        project_ids=list(project_ids),
        projects={pid: {"config": []} for pid in project_ids},
        backend_params={},
        limit=10,
        threshold=0.0,
        documents=[[document_identity(doc) for doc in batch] for batch in doc_batches],
    )


def _gold_array(batch):
    return csr_array(
        ([True], ([0], [batch[0].subject_set[0]])),
        shape=(1, 2),
        dtype=bool,
    )


def test_run_suggest_batches_submits_in_original_order():
    doc_batches = _make_batches()
    psmap = FakeProjectSuggestMap()

    with multiprocessing.dummy.Pool(1) as pool:
        outcomes = list(annif.parallel.run_suggest_batches(psmap, doc_batches, pool))

    # outcomes are yielded in original order even when computed in parallel
    assert [outcome.index for outcome in outcomes] == [0, 1, 2]
    assert [outcome.documents for outcome in outcomes] == [
        [document_identity(doc) for doc in batch] for batch in doc_batches
    ]
    # workers were also submitted in original batch order
    called_texts = [text for _, text in psmap.calls]
    assert called_texts.index("d0") < called_texts.index("d1")
    assert called_texts.index("d1") < called_texts.index("d2")
    assert all(not outcome.failures for outcome in outcomes)
    assert all(not outcome.cached for outcome in outcomes)


def test_run_suggest_batches_isolates_failing_project():
    doc_batches = _make_batches()
    # project p1 fails on every batch, p2 keeps working
    psmap = FakeProjectSuggestMap(fail_projects={"p1"})

    with multiprocessing.dummy.Pool(1) as pool:
        outcomes = list(annif.parallel.run_suggest_batches(psmap, doc_batches, pool))

    for outcome in outcomes:
        assert "p1" in outcome.failures
        assert "p2" not in outcome.failures
        # the other project's suggestions are retained and aligned with
        # the correct document
        assert set(outcome.hit_sets.keys()) == {"p2"}
        assert len(outcome.hit_sets["p2"]) == 1
        expected_score = float(outcome.index) + 1.1
        assert outcome.hit_sets["p2"].array[0, 0] == np.float32(expected_score)


def test_run_suggest_batches_survives_catastrophic_worker_error():
    doc_batches = _make_batches()
    psmap = ExplodingSuggestMap()

    with multiprocessing.dummy.Pool(1) as pool:
        outcomes = list(annif.parallel.run_suggest_batches(psmap, doc_batches, pool))

    # the pool is not terminated: every batch gets an outcome
    assert len(outcomes) == 3
    for outcome in outcomes:
        assert outcome.hit_sets is None
        assert set(outcome.failures.keys()) == {"p1", "p2"}
        assert "catastrophic failure" in outcome.failures["p1"]


def test_run_suggest_batches_resumes_from_checkpoint_without_reinferring(tmpdir):
    doc_batches = _make_batches()
    fingerprint = _make_fingerprint(doc_batches)
    checkpoint = BatchCheckpoint.open(str(tmpdir), fingerprint)

    # first run: batch 1 fails for both projects
    psmap = FakeProjectSuggestMap(fail_texts={"d1"})
    with multiprocessing.dummy.Pool(1) as pool:
        for outcome in annif.parallel.run_suggest_batches(
            psmap, doc_batches, pool, checkpoint
        ):
            if not outcome.failures:
                checkpoint.save_batch(
                    outcome.index,
                    outcome.hit_sets,
                    _gold_array(doc_batches[outcome.index]),
                    checkpoint.owner_id,
                )
            else:
                checkpoint.record_failure(
                    outcome.index,
                    outcome.documents,
                    "p1",
                    outcome.failures["p1"],
                    checkpoint.owner_id,
                )

    assert checkpoint.pending_indices() == [1]
    assert {failure["batch"] for failure in checkpoint.failures} == {1}

    # second run: only the failed batch is inferred again
    psmap2 = FakeProjectSuggestMap()
    with multiprocessing.dummy.Pool(1) as pool:
        outcomes = list(
            annif.parallel.run_suggest_batches(psmap2, doc_batches, pool, checkpoint)
        )

    assert [outcome.index for outcome in outcomes] == [0, 1, 2]
    assert [outcome.cached for outcome in outcomes] == [True, False, True]
    inferred_texts = sorted({text for _, text in psmap2.calls})
    assert inferred_texts == ["d1"]
    # cached batches carry the original suggestions and gold aggregate
    assert outcomes[0].hit_sets["p1"].array[0, 0] == np.float32(1.0)
    assert outcomes[0].hit_sets["p2"].array[0, 0] == np.float32(1.1)
    assert outcomes[0].gold[0, 0]
    assert outcomes[2].gold[0, 0]
