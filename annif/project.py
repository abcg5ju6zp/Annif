"""Project management functionality for Annif"""

from __future__ import annotations

import enum
import hashlib
import json
import os.path
import re
import threading
from shutil import rmtree
from typing import TYPE_CHECKING, Any

import annif
import annif.analyzer
import annif.backend
import annif.corpus
import annif.model
import annif.transform
from annif.corpus import Document
from annif.datadir import DatadirMixin
from annif.exception import (
    AnnifException,
    ConfigurationException,
    NotEnabledException,
    NotInitializedException,
    NotSupportedException,
)
from annif.util import parse_args
from annif.vocab import SubjectIndexFilter, kwargs_to_exclude_uris

if TYPE_CHECKING:
    from collections import defaultdict
    from configparser import SectionProxy
    from datetime import datetime

    from click.utils import LazyFile

    from annif.analyzer import Analyzer
    from annif.backend import AnnifBackend
    from annif.backend.hyperopt import HPRecommendation
    from annif.corpus import DocumentCorpus
    from annif.corpus.subject import SubjectIndex
    from annif.model import ModelGenerationManager
    from annif.registry import AnnifRegistry
    from annif.transform.transform import TransformChain
    from annif.vocab import AnnifVocabulary

logger = annif.logger


class Access(enum.IntEnum):
    """Enumeration of access levels for projects"""

    private = 1
    hidden = 2
    public = 3


class AnnifProject(DatadirMixin):
    """Class representing the configuration of a single Annif project."""

    # defaults for uninitialized instances
    _transform = None
    _analyzer = None
    _backend = None
    _vocab = None
    _vocab_lang = None
    _vocab_kwargs = {}
    _subject_index = None
    _model_manager = None
    initialized = False

    # default values for configuration settings
    DEFAULT_ACCESS = "public"

    def __init__(
        self,
        project_id: str,
        config: dict[str, str] | SectionProxy,
        datadir: str,
        registry: AnnifRegistry,
    ) -> None:
        DatadirMixin.__init__(self, datadir, "projects", project_id)
        self.project_id = project_id
        self.name = config.get("name", project_id)
        self.language = config["language"]
        self.analyzer_spec = config.get("analyzer", None)
        self.transform_spec = config.get("transform", "pass")
        self.vocab_spec = config.get("vocab", None)
        self.config = config
        self._base_datadir = datadir
        self.registry = registry
        self._backend_lock = threading.Lock()
        self._init_access()

    def _init_access(self) -> None:
        access = self.config.get("access", self.DEFAULT_ACCESS)
        try:
            self.access = getattr(Access, access)
        except AttributeError:
            raise ConfigurationException(
                "'{}' is not a valid access setting".format(access),
                project_id=self.project_id,
            )

    def _initialize_analyzer(self) -> None:
        if not self.analyzer_spec:
            return  # not configured, so assume it's not needed
        analyzer = self.analyzer
        logger.debug(
            "Project '%s': initialized analyzer: %s", self.project_id, str(analyzer)
        )

    def _initialize_subjects(self) -> None:
        try:
            subjects = self.subjects
            logger.debug(
                "Project '%s': initialized subjects: %s", self.project_id, str(subjects)
            )
        except AnnifException as err:
            logger.warning(err.format_message())

    def _initialize_backend(self, parallel: bool) -> None:
        logger.debug("Project '%s': initializing backend", self.project_id)
        try:
            if not self.backend:
                logger.debug("Cannot initialize backend: does not exist")
                return
            self.backend.initialize(parallel)
        except AnnifException as err:
            logger.warning(err.format_message())

    def initialize(self, parallel: bool = False) -> None:
        """Initialize this project and its backend so that they are ready to
        be used. If parallel is True, expect that the project will be used
        for parallel processing."""

        if self.initialized:
            return

        logger.debug("Initializing project '%s'", self.project_id)

        self._initialize_analyzer()
        self._initialize_subjects()
        self._initialize_backend(parallel)

        self.initialized = True

    def _suggest_with_backend(
        self,
        docs: list[Document],
        backend: AnnifBackend,
        backend_params: defaultdict[str, dict] | None,
    ) -> annif.suggestion.SuggestionBatch:
        if backend_params is None:
            backend_params = {}
        beparams = backend_params.get(backend.backend_id, {})
        # Use the backend instance captured at the start of the request: a
        # generation switch may replace the cached active instance while the
        # request is in flight, but in-flight requests keep serving with the
        # instance they started with.
        return backend.suggest(docs, beparams)

    @property
    def analyzer(self) -> Analyzer:
        if self._analyzer is None:
            if self.analyzer_spec:
                self._analyzer = annif.analyzer.get_analyzer(self.analyzer_spec)
            else:
                raise ConfigurationException(
                    "analyzer setting is missing", project_id=self.project_id
                )
        return self._analyzer

    @property
    def transform(self) -> TransformChain:
        if self._transform is None:
            self._transform = annif.transform.get_transform(
                self.transform_spec, project=self
            )
        return self._transform

    @property
    def model_manager(self) -> ModelGenerationManager:
        """Return the manager for the on-disk model generations."""
        if self._model_manager is None:
            self._model_manager = annif.model.ModelGenerationManager(self)
        return self._model_manager

    def active_model_dir(self) -> str:
        """Return the directory that holds the active model generation.
        Projects that predate model generations (and untrained projects)
        use the project data directory root."""
        return self.model_manager.active_model_dir()

    @property
    def backend(self) -> AnnifBackend | None:
        if self._backend is None:
            with self._backend_lock:
                if self._backend is None:
                    self._backend = self._create_backend()
        return self._backend

    def _create_backend(self, model_dir: str | None = None) -> AnnifBackend | None:
        """Create a backend instance bound to the given model directory, or
        to the active model generation when no directory is given."""
        if "backend" not in self.config:
            raise ConfigurationException(
                "backend setting is missing", project_id=self.project_id
            )
        backend_id = self.config["backend"]
        try:
            backend_class = annif.backend.get_backend(backend_id)
        except ValueError:
            logger.warning(
                "Could not create backend %s, "
                "make sure you've installed optional dependencies",
                backend_id,
            )
            return None
        if model_dir is None:
            model_dir = (
                self.active_model_dir()
                if backend_class.versioned_model
                else self.datadir
            )
        return backend_class(
            backend_id, config_params=self.config, project=self, datadir=model_dir
        )

    def _initialize_vocab(self) -> None:
        if self.vocab_spec is None:
            raise ConfigurationException(
                "vocab setting is missing", project_id=self.project_id
            )

        match = re.match(r"([\w-]+)(\((.*)\))?$", self.vocab_spec)
        if match is None:
            raise ValueError(f"Invalid vocabulary specification: {self.vocab_spec}")
        vocab_id = match.group(1)
        posargs, self._vocab_kwargs = parse_args(match.group(3))
        self._vocab_lang = posargs[0] if posargs else self.language
        self._vocab = self.registry.get_vocab(vocab_id)

    @property
    def vocab(self) -> AnnifVocabulary:
        if self._vocab is None:
            self._initialize_vocab()
        return self._vocab

    @property
    def vocab_lang(self) -> str:
        if self._vocab_lang is None:
            self._initialize_vocab()
        return self._vocab_lang

    @property
    def subjects(self) -> SubjectIndex:
        if self._subject_index is None:
            self._subject_index = self.vocab.subjects
            exclude_uris = kwargs_to_exclude_uris(self.vocab, self._vocab_kwargs)
            if exclude_uris:
                self._subject_index = SubjectIndexFilter(
                    self._subject_index, exclude=exclude_uris
                )
        return self._subject_index

    def _get_info(self, key: str) -> bool | datetime | None:
        try:
            be = self.backend
            if be is not None:
                return getattr(be, key)
        except AnnifException as err:
            logger.warning(err.format_message())
            return None

    @property
    def is_trained(self) -> bool | None:
        return self._get_info("is_trained")

    @property
    def modification_time(self) -> datetime | None:
        return self._get_info("modification_time")

    def suggest_corpus(
        self,
        corpus: DocumentCorpus,
        backend_params: defaultdict[str, dict] | None = None,
    ) -> annif.suggestion.SuggestionResults:
        """Suggest subjects for the given documents corpus in batches of documents."""
        suggestions = (
            self.suggest(doc_batch, backend_params) for doc_batch in corpus.doc_batches
        )
        import annif.suggestion

        return annif.suggestion.SuggestionResults(suggestions)

    def suggest(
        self,
        documents: list[Document],
        backend_params: defaultdict[str, dict] | None = None,
    ) -> annif.suggestion.SuggestionBatch:
        """Suggest subjects for the given documents batch."""
        backend = self.backend
        try:
            is_trained = backend.is_trained
        except AnnifException as err:
            logger.warning(err.format_message())
            is_trained = None
        if not is_trained:
            if is_trained is None:
                logger.warning("Could not get train state information.")
            else:
                raise NotInitializedException("Project is not trained.")
        transformed_docs = [self.transform.transform_doc(doc) for doc in documents]
        return self._suggest_with_backend(transformed_docs, backend, backend_params)

    def train(
        self,
        corpus: DocumentCorpus,
        backend_params: defaultdict[str, dict] | None = None,
        jobs: int = 0,
    ) -> None:
        """train the project using documents from a metadata source.

        Backends with persistent model files are trained in an isolated
        candidate directory by the unique lock owner, validated, probed and
        published atomically as a new model generation; the active backend
        instance is only switched after the publish succeeds. Backends
        without persistent model files are trained exactly as before."""

        backend = self.backend
        use_cached = corpus == "cached"
        if not use_cached:
            corpus = self.transform.transform_corpus(corpus)
        if backend_params is None:
            backend_params = {}
        beparams = backend_params.get(backend.backend_id, {})
        if not backend.versioned_model:
            backend.train(corpus, beparams, jobs)
            return
        # the parameter version pinned in the manifest is the effective
        # parameter set the backend trains with (defaults + config + CLI)
        effective_params = backend._get_backend_params(beparams)
        self._train_versioned(
            backend, corpus, beparams, effective_params, jobs, use_cached
        )

    def _train_versioned(
        self,
        backend: AnnifBackend,
        corpus: DocumentCorpus | str,
        beparams: dict[str, Any],
        effective_params: dict[str, Any],
        jobs: int,
        use_cached: bool,
    ) -> None:
        """Train a file-backed backend as a new, atomically published model
        generation."""

        manager = self.model_manager
        with manager.ownership("train") as owner:
            parent = manager.active_generation_id()
            if use_cached:
                # the cached training data was produced by a previous run, so
                # its corpus fingerprint remains the one to pin
                parent_manifest = manager.active_manifest()
                if parent_manifest is not None:
                    corpus_info = parent_manifest["fingerprint"]["corpus"]
                else:
                    corpus_info = {"documents": None, "sha256": None, "cached": True}
                manager.seed_from_active(owner.staging_dir, include_train_data=True)
            else:
                corpus_info = annif.model.corpus_fingerprint(corpus)
            candidate = self._create_backend(owner.staging_dir)
            candidate.train(corpus, beparams, jobs)
            self._publish_candidate(
                owner,
                candidate,
                effective_params,
                corpus_info,
                operation="train",
                parent=parent,
            )

    def learn(
        self,
        corpus: DocumentCorpus,
        backend_params: defaultdict[str, dict] | None = None,
    ) -> None:
        """further train the project using documents from a metadata source"""
        if not isinstance(self.backend, annif.backend.backend.AnnifLearningBackend):
            raise NotSupportedException(
                "Learning not supported by backend", project_id=self.project_id
            )
        if not annif.util.boolean(self.config.get("allow_learn", False)):
            raise NotEnabledException(
                "Learning not enabled for project", project_id=self.project_id
            )

        backend = self.backend
        if backend_params is None:
            backend_params = {}
        beparams = backend_params.get(backend.backend_id, {})
        corpus = self.transform.transform_corpus(corpus)
        if not backend.versioned_model:
            backend.learn(corpus, beparams)
            return
        effective_params = backend._get_backend_params(beparams)
        self._learn_versioned(backend, corpus, effective_params)

    def _learn_versioned(
        self,
        backend: AnnifBackend,
        corpus: DocumentCorpus,
        effective_params: dict[str, Any],
    ) -> None:
        """Further train a file-backed backend by seeding a candidate from
        the active generation and publishing the result as a new
        generation."""

        manager = self.model_manager
        with manager.ownership("learn") as owner:
            parent = manager.active_generation_id()
            manager.seed_from_active(owner.staging_dir, include_train_data=False)
            candidate = self._create_backend(owner.staging_dir)
            corpus_info = annif.model.corpus_fingerprint(corpus)
            candidate.learn(corpus, effective_params)
            self._publish_candidate(
                owner,
                candidate,
                effective_params,
                corpus_info,
                operation="learn",
                parent=parent,
            )

    def _publish_candidate(
        self,
        owner,
        candidate: AnnifBackend,
        beparams: dict[str, Any],
        corpus_info: dict[str, Any],
        operation: str,
        parent: str | None,
    ) -> None:
        """Validate, probe and atomically publish a trained candidate
        generation, then switch the cached active backend instance."""

        manager = self.model_manager
        files = manager.finalize_candidate(owner.staging_dir)
        # Probe with a separate instance that loads the written files cold,
        # simulating how the generation will be used after a restart.
        probe = self._create_backend(owner.staging_dir)
        annif.model.probe_backend(probe, beparams)
        fingerprint = self._model_fingerprint(beparams, corpus_info)
        generation_id = manager.new_generation_id(fingerprint["sha256"], owner.token)
        manager.write_manifest(
            owner.staging_dir,
            generation_id=generation_id,
            operation=operation,
            parent=parent,
            fingerprint=fingerprint,
            files=files,
        )
        manager.commit(owner, generation_id)
        previous = parent
        manager.publish_pointer(generation_id, previous)
        try:
            activated = self._create_backend(manager.generation_dir(generation_id))
            activated.initialize()
        except AnnifException as err:
            # files were published, but the live instance cannot use them:
            # roll the active generation back and keep diagnostics
            manager.rollback_after_activation_failure(generation_id, previous, err)
            raise
        with self._backend_lock:
            self._backend = activated
        manager.prune_generations()
        logger.info(
            "Project '%s': published model generation %s via %s",
            self.project_id,
            generation_id,
            operation,
        )

    def _model_fingerprint(
        self, beparams: dict[str, Any], corpus_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the fingerprint pinning the analyzer, vocabulary, backend
        parameter and corpus versions of a model generation."""

        analyzer_info: dict[str, Any] = {"spec": self.analyzer_spec}
        try:
            analyzer = self.analyzer
            analyzer_info.update(
                {
                    "name": getattr(analyzer, "name", None),
                    "param": getattr(analyzer, "param", None),
                    "token_min_length": getattr(analyzer, "token_min_length", None),
                }
            )
        except AnnifException:
            pass

        vocab_info: dict[str, Any] = {
            "spec": self.vocab_spec,
            "id": None,
            "language": self.vocab_lang,
            "kwargs": dict(sorted(self._vocab_kwargs.items())),
        }
        try:
            vocab = self.vocab
            vocab_info["id"] = vocab.vocab_id
            vocab_info["subjects"] = len(vocab.subjects)
            subjects_path = os.path.join(vocab.datadir, vocab.INDEX_FILENAME_CSV)
            if os.path.exists(subjects_path):
                stat = os.stat(subjects_path)
                vocab_info["subjects_file"] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
        except AnnifException:
            pass

        backend_info = {
            "id": self.config.get("backend"),
            "params": {key: beparams[key] for key in sorted(beparams)},
        }
        components = {
            "analyzer": analyzer_info,
            "vocab": vocab_info,
            "backend": backend_info,
            "corpus": corpus_info,
        }
        canonical = json.dumps(
            components, sort_keys=True, default=str, ensure_ascii=False
        )
        components["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return components

    def hyperopt(
        self,
        corpus: DocumentCorpus,
        trials: int,
        jobs: int,
        metric: str,
        results_file: LazyFile | None,
    ) -> HPRecommendation:
        """optimize the hyperparameters of the project using a validation
        corpus against a given metric"""
        if isinstance(self.backend, annif.backend.hyperopt.AnnifHyperoptBackend):
            optimizer = self.backend.get_hp_optimizer(corpus, metric)
            return optimizer.optimize(trials, jobs, results_file)

        raise NotSupportedException(
            "Hyperparameter optimization not supported " "by backend",
            project_id=self.project_id,
        )

    def dump(self) -> dict[str, str | dict | bool | datetime | None]:
        """return this project as a dict"""

        try:
            vocab = self.vocab.dump()
            vocab_lang = self.vocab_lang
        except ConfigurationException:
            vocab = None
            vocab_lang = None

        return {
            "project_id": self.project_id,
            "name": self.name,
            "language": self.language,
            "backend": {"backend_id": self.config.get("backend")},
            "vocab": vocab,
            "vocab_language": vocab_lang,
            "is_trained": self.is_trained,
            "modification_time": self.modification_time,
        }

    def remove_model_data(self) -> None:
        """remove the data of this project"""
        datadir_path = self._datadir_path
        if os.path.isdir(datadir_path):
            rmtree(datadir_path)
            with self._backend_lock:
                self._backend = None
            self._model_manager = None
            logger.info("Removed model data for project {}.".format(self.project_id))
        else:
            logger.warning(
                "No model data to remove for project {}.".format(self.project_id)
            )
