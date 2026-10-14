"""Support for document corpora in JSON format"""

import functools
import json
import os.path
from importlib.resources import files

import annif
from annif.vocab import SubjectIndex

from .types import Document, SubjectSet

logger = annif.logger


@functools.lru_cache(maxsize=1)
def _get_json_schema(schema_name):
    schema_path = files("annif.schemas").joinpath(schema_name)
    with schema_path.open("r", encoding="utf-8") as schema_file:
        return json.load(schema_file)


def json_to_document(
    filename: str,
    json_data: str,
    subject_index: SubjectIndex | None,
    language: str,
    require_subjects: bool,
    retain_subjectless: bool = False,
) -> Document | None:
    """parse a JSON/JSONL document string. When retain_subjectless is True,
    documents without any scorable subjects are retained (with the evidence of
    why their annotations could not be used) instead of being dropped, so that
    evaluation entry points can account for them consistently."""

    import jsonschema

    try:
        data = json.loads(json_data)
    except json.JSONDecodeError as err:
        logger.warning(f"JSON parsing failed for file {filename}: {err}")
        return None

    try:
        jsonschema.validate(instance=data, schema=_get_json_schema("document.json"))
    except jsonschema.ValidationError as err:
        logger.warning(f"JSON validation failed for file {filename}: {err.message}")
        return None

    if require_subjects or (subject_index is not None and "subjects" in data):
        subject_set = SubjectSet.from_subject_dicts(
            data.get("subjects", []), subject_index, language
        )
        if not subject_set.is_scorable and not retain_subjectless:
            return None
    else:
        subject_set = None

    return Document(
        text=data.get("text", ""),
        metadata=data.get("metadata", {}),
        subject_set=subject_set,
        file_path=filename,
        document_id=data.get("document_id", None),
    )


def json_file_to_document(
    filename: str,
    subject_index: SubjectIndex | None,
    language: str,
    require_subjects: bool,
    retain_subjectless: bool = False,
) -> Document | None:
    if os.path.getsize(filename) == 0:
        logger.warning(f"Skipping empty file {filename}")
        return None

    with open(filename, "r", encoding="utf-8") as jsonfile:
        json_data = jsonfile.read()

    return json_to_document(
        filename,
        json_data,
        subject_index,
        language,
        require_subjects,
        retain_subjectless,
    )
