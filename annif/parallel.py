"""Parallel processing functionality for Annif"""

from __future__ import annotations

import multiprocessing
import multiprocessing.dummy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from annif.checkpoint import BatchCheckpoint, document_identity

if TYPE_CHECKING:
    from collections import defaultdict
    from collections.abc import Iterator
    from typing import Callable

    from scipy.sparse import csr_array

    from annif.corpus import Document, SubjectSet
    from annif.registry import AnnifRegistry
    from annif.suggestion import SuggestionBatch, SuggestionResult


# Start method for processes created by the multiprocessing module.
# A value of None means using the platform-specific default.
# Intended to be overridden in unit tests.
MP_START_METHOD = None


class BaseWorker:
    """Base class for workers that implement tasks executed via
    multiprocessing. The init method can be used to store data objects that
    are necessary for the operation. They will be stored in a class
    attribute that is accessible to the static worker method. The storage
    solution is inspired by this blog post:
    https://thelaziestprogrammer.com/python/multiprocessing-pool-a-global-solution # noqa
    """

    args = None

    @classmethod
    def init(cls, args) -> None:
        cls.args = args  # pragma: no cover


class ProjectSuggestMap:
    """A utility class that can be used to wrap one or more projects and
    provide a mapping method that converts Document objects to suggestions.
    Intended to be used with the multiprocessing module."""

    def __init__(
        self,
        registry: AnnifRegistry,
        project_ids: list[str],
        backend_params: defaultdict[str, Any] | None,
        limit: int | None,
        threshold: float,
    ) -> None:
        self.registry = registry
        self.project_ids = project_ids
        self.backend_params = backend_params
        self.limit = limit
        self.threshold = threshold

    def suggest(self, doc: Document) -> tuple[dict[str, SuggestionResult], SubjectSet]:
        filtered_hits = {}
        for project_id in self.project_ids:
            project = self.registry.get_project(project_id)
            batch = project.suggest([doc], self.backend_params)
            filtered_hits[project_id] = batch.filter(self.limit, self.threshold)[0]
        return (filtered_hits, doc.subject_set)

    def suggest_batch(
        self, batch
    ) -> tuple[dict[str, SuggestionBatch], Iterator[SubjectSet]]:
        filtered_hit_sets = {}
        subject_sets = [doc.subject_set for doc in batch]

        for project_id in self.project_ids:
            project = self.registry.get_project(project_id)
            suggestion_batch = project.suggest(batch, self.backend_params)
            filtered_hit_sets[project_id] = suggestion_batch.filter(
                self.limit, self.threshold
            )
        return (filtered_hit_sets, subject_sets)

    def suggest_batch_isolated(
        self, batch: list[Document]
    ) -> tuple[dict[str, SuggestionBatch], list[SubjectSet], dict[str, str]]:
        """Suggest for a batch like suggest_batch, but never let one
        project's failure kill the whole batch: suggestions of the other
        projects are returned normally and per-project failures are
        reported in the third return value instead of being raised. This
        keeps a single failed project from misaligning or discarding the
        suggestions and metrics of the other projects."""

        filtered_hit_sets: dict[str, SuggestionBatch] = {}
        failures: dict[str, str] = {}
        subject_sets = [doc.subject_set for doc in batch]

        for project_id in self.project_ids:
            try:
                project = self.registry.get_project(project_id)
                suggestion_batch = project.suggest(batch, self.backend_params)
                filtered_hit_sets[project_id] = suggestion_batch.filter(
                    self.limit, self.threshold
                )
            except Exception as err:  # noqa: BLE001 - isolate projects in workers
                failures[project_id] = f"{type(err).__name__}: {err}"
        return filtered_hit_sets, subject_sets, failures


@dataclass
class BatchOutcome:
    """The result of processing one document batch.

    index is the position of the batch in the original input order,
    documents its fixed document identities, hit_sets the per-project
    filtered suggestions, subject_sets the gold subject sets (for fresh
    batches) and gold the gold subjects as a sparse array (for batches
    replayed from a checkpoint). Failures is a mapping of project ID to
    error message for projects whose inference failed."""

    index: int
    documents: list[list]
    hit_sets: dict[str, SuggestionBatch] | None
    subject_sets: list[SubjectSet] | None = None
    gold: csr_array | None = None
    failures: dict[str, str] = field(default_factory=dict)
    cached: bool = False


def suggest_batch_task(
    task: tuple[ProjectSuggestMap, int, list[list], list[Document]],
) -> BatchOutcome:
    """Module-level worker function (picklable also with the spawn start
    method) that suggests subjects for one batch, returning project
    failures as data rather than raising so that one failed worker can no
    longer terminate the whole pool."""

    psmap, batch_index, identities, docs = task
    try:
        hit_sets, subject_sets, failures = psmap.suggest_batch_isolated(docs)
    except Exception as err:  # noqa: BLE001 - keep the pool alive
        return BatchOutcome(
            index=batch_index,
            documents=identities,
            hit_sets=None,
            failures={
                project_id: f"{type(err).__name__}: {err}"
                for project_id in psmap.project_ids
            },
        )
    return BatchOutcome(
        index=batch_index,
        documents=identities,
        hit_sets=hit_sets,
        subject_sets=subject_sets,
        failures=failures,
    )


def run_suggest_batches(
    psmap: ProjectSuggestMap,
    doc_batches: list[list[Document]],
    pool,
    checkpoint: BatchCheckpoint | None = None,
) -> Iterator[BatchOutcome]:
    """Run suggestion for all document batches via the given pool and
    yield a BatchOutcome for every batch in the original input order.

    Batches are submitted to workers in original order. Batches already
    committed to the checkpoint are not submitted at all; they are
    replayed from the checkpoint, so no model inference is repeated.
    Worker or project failures are reported on the outcome instead of
    aborting the pool, and committed batches keep their stable boundary."""

    identities = [[document_identity(doc) for doc in batch] for batch in doc_batches]
    if checkpoint is None:
        pending = list(range(len(doc_batches)))
    else:
        pending = checkpoint.pending_indices()
    pending_set = set(pending)

    tasks = (
        (psmap, batch_index, identities[batch_index], doc_batches[batch_index])
        for batch_index in pending
    )
    computed = iter(pool.imap(suggest_batch_task, tasks))

    for batch_index in range(len(doc_batches)):
        if batch_index in pending_set:
            yield next(computed)
        else:
            hit_sets, gold = checkpoint.load_batch(batch_index)
            yield BatchOutcome(
                index=batch_index,
                documents=identities[batch_index],
                hit_sets=hit_sets,
                gold=gold,
                cached=True,
            )


def get_pool(n_jobs: int) -> tuple[int | None, Callable]:
    """return a suitable constructor for multiprocessing pool class, and the correct
    jobs argument for it, for the given amount of parallel jobs"""

    ctx = multiprocessing.get_context(MP_START_METHOD)

    if n_jobs < 1:
        n_jobs = None
        pool_constructor: Callable = ctx.Pool
    elif n_jobs == 1:
        # use the dummy wrapper around threading to avoid subprocess overhead
        pool_constructor = multiprocessing.dummy.Pool
    else:
        pool_constructor = ctx.Pool

    return n_jobs, pool_constructor
