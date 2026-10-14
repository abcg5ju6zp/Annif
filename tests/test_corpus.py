"""Unit tests for corpus functionality in Annif"""

import gzip
import json
import logging

import numpy as np
import pytest

import annif.corpus
import annif.vocab
from annif.corpus import (
    Document,
    GoldCorpusCoverage,
    ScorableDocumentCorpus,
    SubjectSet,
    TransformingDocumentCorpus,
)
from annif.exception import NotSupportedException, OperationFailedException


def test_document(subject_index):
    doc = Document(text="Hello world")
    assert doc.text == "Hello world"
    assert doc.subject_set == SubjectSet()
    assert doc.metadata == {}
    assert doc.file_path is None
    assert repr(doc) == (
        "Document(document_id=None, text='Hello world', subject_set=SubjectSet([]), "
        "metadata={}, file_path=None)"
    )
    assert doc.as_dict(subject_index, "en") == {"text": "Hello world"}


def test_document_with_all_fields(subject_index):
    doc = Document(
        text="Hello world",
        document_id="test",
        subject_set=SubjectSet([42, 55]),
        metadata={"title": "Hello"},
        file_path="hello.txt",
    )
    assert doc.text == "Hello world"
    assert doc.document_id == "test"
    assert doc.subject_set == SubjectSet([42, 55])
    assert doc.metadata == {"title": "Hello"}
    assert doc.file_path == "hello.txt"
    assert repr(doc) == (
        "Document(document_id='test', text='Hello world', "
        "subject_set=SubjectSet([42, 55]), metadata={'title': 'Hello'}, "
        "file_path='hello.txt')"
    )
    assert doc.as_dict(subject_index, "en") == {
        "document_id": "test",
        "metadata": {
            "title": "Hello",
        },
        "subjects": [
            {
                "label": "ancient DNA",
                "uri": "http://www.yso.fi/onto/yso/p29546",
            },
            {
                "label": "Megalithic culture",
                "uri": "http://www.yso.fi/onto/yso/p14800",
            },
        ],
        "text": "Hello world",
    }


def test_subjectset_uris(subject_index):
    data = """<http://www.yso.fi/onto/yso/p2558>\trautakausi
    <http://www.yso.fi/onto/yso/p12738>\tviikinkiaika
    """

    sset = annif.corpus.SubjectSet.from_string(data, subject_index, "fi")
    assert len(sset) == 2
    assert subject_index.by_uri("http://www.yso.fi/onto/yso/p2558") in sset
    assert subject_index.by_uri("http://www.yso.fi/onto/yso/p12738") in sset


def test_subjectset_labels(subject_index):
    data = """rautakausi
    viikinkiaika
    """

    sset = annif.corpus.SubjectSet.from_string(data, subject_index, "fi")
    assert len(sset) == 2
    assert subject_index.by_label("rautakausi", "fi") in sset
    assert subject_index.by_label("viikinkiaika", "fi") in sset


def test_subjectset_from_list(subject_index):
    uris = ["http://www.yso.fi/onto/yso/p10849", "http://www.yso.fi/onto/yso/p19740"]
    subject_ids = [subject_index.by_uri(uri) for uri in uris]
    sset = annif.corpus.SubjectSet(subject_ids)
    assert len(sset) == 2
    assert subject_index.by_uri("http://www.yso.fi/onto/yso/p10849") in sset
    assert subject_index.by_uri("http://www.yso.fi/onto/yso/p19740") in sset


def test_subjectset_empty():
    sset = annif.corpus.SubjectSet()
    assert len(sset) == 0
    assert not sset
    with pytest.raises(IndexError):
        sset[0]


def test_subjectset_equal():
    sset = annif.corpus.SubjectSet([1, 3, 5])
    sset2 = annif.corpus.SubjectSet([3, 5, 1])
    assert sset == sset2


def test_subjectset_nonequal():
    sset = annif.corpus.SubjectSet([1, 3, 5])
    sset2 = annif.corpus.SubjectSet([3, 5])
    assert sset != sset2
    assert sset != [1, 3, 5]
    assert sset != set([1, 3, 5])


def test_subjectset_as_vector(subject_index):
    uris = ["http://www.yso.fi/onto/yso/p10849", "http://example.org/unknown"]
    subject_ids = [subject_index.by_uri(uri) for uri in uris]
    sset = annif.corpus.SubjectSet(subject_ids)
    vector = sset.as_vector(len(subject_index))
    assert vector.sum() == 1  # only one known subject


def test_subjectset_as_vector_destination(subject_index):
    uris = ["http://www.yso.fi/onto/yso/p10849", "http://example.org/unknown"]
    subject_ids = [subject_index.by_uri(uri) for uri in uris]
    sset = annif.corpus.SubjectSet(subject_ids)
    destination = np.zeros(len(subject_index), dtype=np.float32)
    vector = sset.as_vector(destination=destination)
    assert vector.sum() == 1  # only one known subject
    assert vector is destination


def test_docdir_key(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("key1")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("key2")
    tmpdir.join("doc3.txt").write("doc3")

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), require_subjects=False)
    files = sorted(docdir)
    assert len(files) == 3
    assert files[0] == str(tmpdir.join("doc1.txt"))
    assert files[1] == str(tmpdir.join("doc2.txt"))
    assert files[2] == str(tmpdir.join("doc3.txt"))


def test_docdir_tsv(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.tsv").write("<http://example.org/key1>\tkey1")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.tsv").write("<http://example.org/key2>\tkey2")
    tmpdir.join("doc3.txt").write("doc3")

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), require_subjects=False)
    files = sorted(docdir)
    assert len(files) == 3
    assert files[0] == str(tmpdir.join("doc1.txt"))
    assert files[1] == str(tmpdir.join("doc2.txt"))
    assert files[2] == str(tmpdir.join("doc3.txt"))


def test_docdir_tsv_bom(tmpdir, subject_index):
    tmpdir.join("doc1.txt").write("doc1".encode("utf-8-sig"))
    tmpdir.join("doc1.tsv").write(
        "<http://www.yso.fi/onto/yso/p4622>\tesihistoria".encode("utf-8-sig")
    )
    tmpdir.join("doc2.txt").write("doc2".encode("utf-8-sig"))
    tmpdir.join("doc2.tsv").write(
        "<http://www.yso.fi/onto/yso/p2558>\trautakausi".encode("utf-8-sig")
    )

    docdir = annif.corpus.DocumentDirectory(
        str(tmpdir), subject_index, "fi", require_subjects=True
    )
    docs = list(docdir.documents)
    assert docs[0].text == "doc1"
    assert (
        subject_index.by_uri("http://www.yso.fi/onto/yso/p4622") in docs[0].subject_set
    )
    assert len(docs[0].subject_set) == 1
    assert docs[1].text == "doc2"
    assert (
        subject_index.by_uri("http://www.yso.fi/onto/yso/p2558") in docs[1].subject_set
    )
    assert len(docs[1].subject_set) == 1


def test_docdir_json(tmpdir):
    data1 = {"text": "doc1", "subjects": [{"uri": "http://example.org/key1"}]}
    tmpdir.join("doc1.json").write(json.dumps(data1))
    data2 = {"text": "doc2", "subjects": [{"uri": "http://example.org/key2"}]}
    tmpdir.join("doc2.json").write(json.dumps(data2))
    data3 = {"text": "doc3"}
    tmpdir.join("doc3.json").write(json.dumps(data3))

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), require_subjects=False)
    files = sorted(docdir)
    assert len(files) == 3
    assert files[0] == str(tmpdir.join("doc1.json"))
    assert files[1] == str(tmpdir.join("doc2.json"))
    assert files[2] == str(tmpdir.join("doc3.json"))


def test_docdir_json_empty(tmpdir, caplog):
    tmpdir.join("doc1.json").write("")

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), require_subjects=False)
    files = sorted(docdir)
    assert len(files) == 1
    assert files[0] == str(tmpdir.join("doc1.json"))

    docs = list(docdir.documents)
    assert len(docs) == 0

    assert len(docs) == 0
    assert "Skipping empty file" in caplog.text


def test_docdir_json_broken(tmpdir, caplog):
    tmpdir.join("doc1.json").write('{ "broken_syntax": yes }')

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), require_subjects=False)
    files = sorted(docdir)
    assert len(files) == 1
    assert files[0] == str(tmpdir.join("doc1.json"))

    with caplog.at_level(logging.WARNING):
        docs = list(docdir.documents)

    assert len(docs) == 0
    assert "JSON parsing failed" in caplog.text


def test_docdir_json_invalid(tmpdir, caplog):
    tmpdir.join("doc1.json").write(json.dumps({"follows_schema": False}))

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), require_subjects=False)
    files = sorted(docdir)
    assert len(files) == 1
    assert files[0] == str(tmpdir.join("doc1.json"))

    with caplog.at_level(logging.WARNING):
        docs = list(docdir.documents)

    assert len(docs) == 0
    assert "JSON validation failed" in caplog.text


def test_docdir_key_require_subjects(tmpdir, subject_index):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("<http://example.org/key1>\tkey1")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("<http://example.org/key2>\tkey2")
    tmpdir.join("doc3.txt").write("doc3")

    docdir = annif.corpus.DocumentDirectory(
        str(tmpdir), subject_index, "en", require_subjects=True
    )

    # the docdir contains 3 files
    files = sorted(docdir)
    assert len(files) == 3
    assert files[0] == str(tmpdir.join("doc1.txt"))
    assert files[1] == str(tmpdir.join("doc2.txt"))
    assert files[2] == str(tmpdir.join("doc3.txt"))

    # only 2 of the files include subjects
    docs = list(docdir.documents)
    assert len(docs) == 2


def test_docdir_tsv_require_subjects(tmpdir, subject_index):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.tsv").write("key1")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.tsv").write("key2")
    tmpdir.join("doc3.txt").write("doc3")

    docdir = annif.corpus.DocumentDirectory(
        str(tmpdir), subject_index, "en", require_subjects=True
    )

    # the docdir contains 3 files
    files = sorted(docdir)
    assert len(files) == 3
    assert files[0] == str(tmpdir.join("doc1.txt"))
    assert files[1] == str(tmpdir.join("doc2.txt"))
    assert files[2] == str(tmpdir.join("doc3.txt"))

    # only 2 of the files include subjects
    docs = list(docdir.documents)
    assert len(docs) == 2


def test_docdir_json_require_subjects(tmpdir, subject_index):
    data1 = {"text": "doc1", "subjects": [{"uri": "http://www.yso.fi/onto/yso/p2558"}]}
    tmpdir.join("doc1.json").write(json.dumps(data1))
    data2 = {"text": "doc2", "subjects": [{"label": "prehistory"}]}
    tmpdir.join("doc2.json").write(json.dumps(data2))
    data3 = {"text": "doc3"}
    tmpdir.join("doc3.json").write(json.dumps(data3))

    docdir = annif.corpus.DocumentDirectory(
        str(tmpdir), subject_index, "en", require_subjects=True
    )

    # the docdir contains 3 files
    files = sorted(docdir)
    assert len(files) == 3
    assert files[0] == str(tmpdir.join("doc1.json"))
    assert files[1] == str(tmpdir.join("doc2.json"))
    assert files[2] == str(tmpdir.join("doc3.json"))

    # only 2 of the files include subjects
    docs = list(docdir.documents)
    assert len(docs) == 2
    assert docs[0].subject_set == SubjectSet(
        [subject_index.by_uri("http://www.yso.fi/onto/yso/p2558")]
    )
    assert docs[1].subject_set == SubjectSet(
        [subject_index.by_uri("http://www.yso.fi/onto/yso/p4622")]
    )


def test_docdir_tsv_as_doccorpus(tmpdir, subject_index):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.tsv").write("<http://www.yso.fi/onto/yso/p4622>\tesihistoria")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.tsv").write("<http://www.yso.fi/onto/yso/p2558>\trautakausi")
    tmpdir.join("doc3.txt").write("doc3")

    docdir = annif.corpus.DocumentDirectory(
        str(tmpdir), subject_index, "fi", require_subjects=True
    )
    docs = list(docdir.documents)
    assert len(docs) == 2
    assert docs[0].text == "doc1"
    assert len(docs[0].subject_set) == 1
    assert (
        subject_index.by_uri("http://www.yso.fi/onto/yso/p4622") in docs[0].subject_set
    )
    assert docs[1].text == "doc2"
    assert (
        subject_index.by_uri("http://www.yso.fi/onto/yso/p2558") in docs[1].subject_set
    )
    assert len(docs[1].subject_set) == 1


def test_docdir_json_as_doccorpus(tmpdir, subject_index):
    data1 = {"text": "doc1", "subjects": [{"uri": "http://www.yso.fi/onto/yso/p4622"}]}
    tmpdir.join("doc1.json").write(json.dumps(data1))
    data2 = {"text": "doc2", "subjects": [{"uri": "http://www.yso.fi/onto/yso/p2558"}]}
    tmpdir.join("doc2.json").write(json.dumps(data2))
    data3 = {"text": "doc3"}
    tmpdir.join("doc3.json").write(json.dumps(data3))

    docdir = annif.corpus.DocumentDirectory(
        str(tmpdir), subject_index, "fi", require_subjects=True
    )
    docs = list(docdir.documents)
    assert len(docs) == 2
    assert docs[0].text == "doc1"
    assert len(docs[0].subject_set) == 1
    assert (
        subject_index.by_uri("http://www.yso.fi/onto/yso/p4622") in docs[0].subject_set
    )
    assert docs[1].text == "doc2"
    assert (
        subject_index.by_uri("http://www.yso.fi/onto/yso/p2558") in docs[1].subject_set
    )
    assert len(docs[1].subject_set) == 1


def test_docdir_key_as_doccorpus(tmpdir, subject_index):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("arkeologit")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("kalliotaide")
    tmpdir.join("doc3.txt").write("doc3")

    docdir = annif.corpus.DocumentDirectory(
        str(tmpdir), subject_index, "fi", require_subjects=True
    )
    docs = list(docdir.documents)
    assert len(docs) == 2
    assert docs[0].text == "doc1"
    assert len(docs[0].subject_set) == 1
    assert (
        subject_index.by_uri("http://www.yso.fi/onto/yso/p10849") in docs[0].subject_set
    )
    assert docs[1].text == "doc2"
    assert len(docs[1].subject_set) == 1
    assert (
        subject_index.by_uri("http://www.yso.fi/onto/yso/p13027") in docs[1].subject_set
    )


def test_docfile_tsv_plain(tmpdir, subject_index):
    docfile = tmpdir.join("documents.tsv")
    docfile.write(
        """Läntinen\t<http://www.yso.fi/onto/yso/p2557>
        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>"""
    )

    docs = annif.corpus.DocumentFileTSV(str(docfile), subject_index)
    assert len(list(docs.documents)) == 3


def test_docfile_csv_plain(tmpdir, subject_index):
    docfile = tmpdir.join("documents.csv")
    lines = (
        "text,subject_uris",
        "Läntinen,<http://www.yso.fi/onto/yso/p2557>",
        "Oulunlinnan,<http://www.yso.fi/onto/yso/p7346>",
        '"Harald Hirmuinen",<http://www.yso.fi/onto/yso/p6479>',
    )
    docfile.write("\n".join(lines))

    docs = annif.corpus.DocumentFileCSV(str(docfile), subject_index)
    assert len(list(docs.documents)) == 3


def test_docfile_csv_document_id(tmpdir, subject_index):
    docfile = tmpdir.join("documents.csv")
    lines = (
        "document_id,text,subject_uris",
        "L,Läntinen,<http://www.yso.fi/onto/yso/p2557>",
        "O,Oulunlinnan,<http://www.yso.fi/onto/yso/p7346>",
        'HH,"Harald Hirmuinen",<http://www.yso.fi/onto/yso/p6479>',
    )
    docfile.write("\n".join(lines))

    docs = annif.corpus.DocumentFileCSV(str(docfile), subject_index)
    assert len(list(docs.documents)) == 3
    firstdoc = next(docs.documents)
    assert firstdoc.document_id == "L"


def test_docfile_csv_metadata(tmpdir, subject_index):
    docfile = tmpdir.join("documents-metadata.csv")
    lines = (
        "text,title,subject_uris,author",
        (
            "Consider a future device...,"
            "As We May Think,"
            "http://www.yso.fi/onto/yso/p817,"
            '"Bush, Vannevar"'
        ),
    )
    docfile.write("\n".join(lines))

    docs = annif.corpus.DocumentFileCSV(str(docfile), subject_index)
    firstdoc = next(docs.documents)
    assert firstdoc.text == "Consider a future device..."
    assert firstdoc.metadata["title"] == "As We May Think"
    assert firstdoc.metadata["author"] == "Bush, Vannevar"


def test_docfile_jsonl_plain(tmpdir, subject_index):
    docfile = tmpdir.join("documents.jsonl")
    lines = (
        '{"text": "Läntinen", "document_id": "L", '
        + '"subjects": [{"uri": "http://www.yso.fi/onto/yso/p2557"}]}',
        '{"text": "Oulunlinnan", '
        + '"subjects": [{"uri": "http://www.yso.fi/onto/yso/p7346"}]}',
        '{"text": "Harald Hirmuinen", '
        + '"subjects": [{"uri": "http://www.yso.fi/onto/yso/p6479"}]}',
    )
    docfile.write("\n".join(lines))

    docs = annif.corpus.DocumentFileJSONL(str(docfile), subject_index, "fi")
    assert len(list(docs.documents)) == 3
    firstdoc = next(docs.documents)
    assert firstdoc.document_id == "L"


def test_docfile_jsonl_broken(tmpdir, subject_index, caplog):
    docfile = tmpdir.join("documents.jsonl")
    docfile.write('{ "broken_syntax": yes }')

    corpus = annif.corpus.DocumentFileJSONL(str(docfile), subject_index, "fi")
    with caplog.at_level(logging.WARNING):
        docs = list(corpus.documents)

    assert len(docs) == 0
    assert "JSON parsing failed" in caplog.text


def test_docfile_jsonl_invalid(tmpdir, subject_index, caplog):
    docfile = tmpdir.join("documents.jsonl")
    docfile.write(json.dumps({"follows_schema": False}))

    corpus = annif.corpus.DocumentFileJSONL(str(docfile), subject_index, "fi")
    with caplog.at_level(logging.WARNING):
        docs = list(corpus.documents)

    assert len(docs) == 0
    assert "JSON validation failed" in caplog.text


def test_docfile_tsv_bom(tmpdir, subject_index):
    docfile = tmpdir.join("documents_bom.tsv")
    data = """Läntinen\t<http://www.yso.fi/onto/yso/p2557>
        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>"""
    docfile.write(data.encode("utf-8-sig"))

    docs = annif.corpus.DocumentFileTSV(str(docfile), subject_index)
    firstdoc = next(docs.documents)
    assert firstdoc.text.startswith("Läntinen")
    assert firstdoc.metadata == {}


def test_docfile_csv_bom(tmpdir, subject_index):
    docfile = tmpdir.join("documents_bom.csv")
    lines = (
        "text,subject_uris",
        "Läntinen,<http://www.yso.fi/onto/yso/p2557>",
        "Oulunlinnan,<http://www.yso.fi/onto/yso/p7346>",
        '"Harald Hirmuinen",<http://www.yso.fi/onto/yso/p6479>',
    )
    docfile.write("\n".join(lines).encode("utf-8-sig"))

    docs = annif.corpus.DocumentFileCSV(str(docfile), subject_index)
    firstdoc = next(docs.documents)
    assert firstdoc.text.startswith("Läntinen")
    assert firstdoc.metadata == {}


def test_docfile_tsv_plain_invalid_lines_require_subjects(
    tmpdir, caplog, subject_index
):
    logger = annif.logger
    logger.propagate = True
    docfile = tmpdir.join("documents_invalid.tsv")
    docfile.write(
        """Läntinen\t<http://www.yso.fi/onto/yso/p2557>

        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        A line with no tabs
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>"""
    )
    docs = annif.corpus.DocumentFileTSV(
        str(docfile), subject_index, require_subjects=True
    )
    assert len(list(docs.documents)) == 3
    assert len(caplog.records) == 2
    expected_msg = "Skipping invalid line (missing tab):"
    for record in caplog.records:
        assert expected_msg in record.message


def test_docfile_tsv_plain_tabless_no_require_subjects(tmpdir, subject_index):
    logger = annif.logger
    logger.propagate = True
    docfile = tmpdir.join("documents_invalid.tsv")
    docfile.write(
        """Läntinen\t<http://www.yso.fi/onto/yso/p2557>

        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        A line with no tabs
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>"""
    )
    corpus = annif.corpus.DocumentFileTSV(
        str(docfile), subject_index, require_subjects=False
    )
    docs = list(corpus.documents)
    assert len(docs) == 5
    assert docs[1].text == ""
    assert docs[3].text == "A line with no tabs"


def test_docfile_csv_plain_invalid_lines(tmpdir, caplog, subject_index):
    docfile = tmpdir.join("documents.csv")
    lines = (
        "text,subject_uris",
        "Läntinen,<http://www.yso.fi/onto/yso/p2557>",
        "Oulunlinnan,<http://www.yso.fi/onto/yso/p7346>",
        "no comma on this line",
        '"Harald Hirmuinen",<http://www.yso.fi/onto/yso/p6479>',
    )
    docfile.write("\n".join(lines))

    docs = annif.corpus.DocumentFileCSV(str(docfile), subject_index)
    assert len(list(docs.documents)) == 4


def test_docfile_csv_plain_invalid_columns_require_subjects(tmpdir, subject_index):
    docfile = tmpdir.join("documents_invalid.csv")
    lines = (
        "text,subject_uri",  # mistyped subject_uris column name
        "Läntinen,<http://www.yso.fi/onto/yso/p2557>",
        "Oulunlinnan,<http://www.yso.fi/onto/yso/p7346>",
        '"Harald Hirmuinen",<http://www.yso.fi/onto/yso/p6479>',
    )
    docfile.write("\n".join(lines).encode("utf-8-sig"))

    docs = annif.corpus.DocumentFileCSV(
        str(docfile), subject_index, require_subjects=True
    )
    with pytest.raises(OperationFailedException) as excinfo:
        list(docs.documents)
    assert str(excinfo.value).startswith("Cannot parse CSV file")


def test_docfile_csv_plain_invalid_columns_no_require_subjects(tmpdir, subject_index):
    docfile = tmpdir.join("documents_invalid.csv")
    lines = (
        "texts,subject_uris",  # mistyped text column name
        "Läntinen,<http://www.yso.fi/onto/yso/p2557>",
        "Oulunlinnan,<http://www.yso.fi/onto/yso/p7346>",
        '"Harald Hirmuinen",<http://www.yso.fi/onto/yso/p6479>',
    )
    docfile.write("\n".join(lines).encode("utf-8-sig"))

    docs = annif.corpus.DocumentFileCSV(
        str(docfile), subject_index, require_subjects=False
    )
    with pytest.raises(OperationFailedException) as excinfo:
        list(docs.documents)
    assert str(excinfo.value).startswith("Cannot parse CSV file")


def test_docfile_csv_plain_no_require_subjects(tmpdir, subject_index):
    docfile = tmpdir.join("documents_invalid.csv")
    lines = (
        "text",
        "Läntinen",
        "Oulunlinnan",
        '"Harald Hirmuinen"',
    )
    docfile.write("\n".join(lines).encode("utf-8-sig"))

    docs = annif.corpus.DocumentFileCSV(
        str(docfile), subject_index, require_subjects=False
    )
    assert len(list(docs.documents)) == 3


def test_docfile_tsv_gzipped(tmpdir, subject_index):
    docfile = tmpdir.join("documents.tsv.gz")
    with gzip.open(str(docfile), "wt") as gzf:
        gzf.write(
            """Pohjoinen\t<http://www.yso.fi/onto/yso/p2557>
            Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
            Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>"""
        )

    docs = annif.corpus.DocumentFileTSV(str(docfile), subject_index)
    assert len(list(docs.documents)) == 3


def test_docfile_csv_gzipped(tmpdir, subject_index):
    docfile = tmpdir.join("documents.csv.gz")
    lines = (
        "text,subject_uris",
        "Läntinen,<http://www.yso.fi/onto/yso/p2557>",
        "Oulunlinnan,<http://www.yso.fi/onto/yso/p7346>",
        '"Harald Hirmuinen",<http://www.yso.fi/onto/yso/p6479>',
    )
    with gzip.open(str(docfile), "wt") as gzf:
        gzf.write("\n".join(lines))

    docs = annif.corpus.DocumentFileCSV(str(docfile), subject_index)
    assert len(list(docs.documents)) == 3


def test_docfile_jsonl_gzipped(tmpdir, subject_index):
    docfile = tmpdir.join("documents.jsonl.gz")
    lines = (
        '{"text": "Läntinen", '
        + '"subjects": [{"uri": "http://www.yso.fi/onto/yso/p2557"}]}',
        '{"text": "Oulunlinnan", '
        + '"subjects": [{"uri": "http://www.yso.fi/onto/yso/p7346"}]}',
        '{"text": "Harald Hirmuinen", '
        + '"subjects": [{"uri": "http://www.yso.fi/onto/yso/p6479"}]}',
    )
    with gzip.open(str(docfile), "wt") as gzf:
        gzf.write("\n".join(lines))

    docs = annif.corpus.DocumentFileJSONL(str(docfile), subject_index, "fi")
    assert len(list(docs.documents)) == 3


def test_docfile_tsv_is_empty(tmpdir, subject_index):
    empty_file = tmpdir.ensure("empty.tsv")
    docs = annif.corpus.DocumentFileTSV(str(empty_file), subject_index)
    assert docs.is_empty()


def test_docfile_csv_is_empty(tmpdir, subject_index):
    empty_file = tmpdir.ensure("empty.csv")
    docs = annif.corpus.DocumentFileCSV(str(empty_file), subject_index)
    with pytest.raises(OperationFailedException) as excinfo:
        list(docs.documents)
    assert str(excinfo.value).startswith("Cannot parse CSV file")


def test_docfile_csv_header_only(tmpdir, subject_index):
    docfile = tmpdir.join("header_only.csv")
    docfile.write("text,subject_uris")
    docs = annif.corpus.DocumentFileCSV(str(docfile), subject_index)
    assert docs.is_empty()


def test_docfile_batches(tmpdir, subject_index):
    docfile = tmpdir.join("documents.tsv")
    docfile.write(
        """Läntinen\t<http://www.yso.fi/onto/yso/p2557>
        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>"""
    )

    docs = annif.corpus.DocumentFileTSV(str(docfile), subject_index)
    docs.DOC_BATCH_SIZE = 1
    assert len(list(docs.doc_batches)) == 3
    assert len(list(docs.doc_batches)[0]) == 1
    docs.DOC_BATCH_SIZE = 3
    assert len(list(docs.doc_batches)) == 1
    assert len(list(docs.doc_batches)[0]) == 3


def test_combinedcorpus(tmpdir, subject_index):
    docfile = tmpdir.join("documents.tsv")
    docfile.write(
        """Läntinen\t<http://www.yso.fi/onto/yso/p2557>
        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>"""
    )

    corpus1 = annif.corpus.DocumentFileTSV(str(docfile), subject_index)
    corpus2 = annif.corpus.DocumentFileTSV(str(docfile), subject_index)

    combined = annif.corpus.CombinedCorpus([corpus1, corpus2])

    assert len(list(combined.documents)) == 6


def test_transformingcorpus(document_corpus):
    def double(doc):
        return Document(doc.text + doc.text, subject_set=doc.subject_set)

    transformed_corpus = TransformingDocumentCorpus(document_corpus, double)
    for transf_doc, doc in zip(transformed_corpus.documents, document_corpus.documents):
        assert transf_doc.text == doc.text + doc.text
        assert transf_doc.subject_set == doc.subject_set
    # Ensure docs are still available after iterating
    assert len(list(transformed_corpus.documents)) == len(
        list(document_corpus.documents)
    )


def test_subjectset_evidence_mixed(subject_index):
    sset = SubjectSet.from_uri_references(
        [
            "http://www.yso.fi/onto/yso/p10849",
            "http://example.org/unknown",
            "http://www.yso.fi/onto/yso/p1747",
        ],
        subject_index,
    )
    assert len(sset) == 2
    assert sset.annotation_count == 3
    assert sset.is_scorable
    assert sset.has_annotations
    assert sset.excluded_annotations == [
        ("http://example.org/unknown", "unknown_uri")
    ]
    assert sset.exclusion_reasons() == {"unknown_uri": 1}


def test_subjectset_evidence_only_unscorable(subject_index):
    sset = SubjectSet.from_uri_references(
        ["http://example.org/unknown1", "http://example.org/unknown2"],
        subject_index,
    )
    assert len(sset) == 0
    assert sset.annotation_count == 2
    assert not sset.is_scorable
    assert sset.has_annotations
    assert sset.excluded_annotations == [
        ("http://example.org/unknown1", "unknown_uri"),
        ("http://example.org/unknown2", "unknown_uri"),
    ]


def test_subjectset_evidence_deprecated():
    index = annif.vocab.SubjectIndexFile()
    index.append(annif.vocab.Subject("http://example.org/deprecated", None, None))
    sset = SubjectSet.from_uri_references(
        ["http://example.org/deprecated"], index
    )
    assert len(sset) == 0
    assert sset.annotation_count == 1
    assert not sset.is_scorable
    assert sset.excluded_annotations == [
        ("http://example.org/deprecated", "deprecated_subject")
    ]


def test_subjectset_evidence_excluded(subject_index):
    subject_filter = annif.vocab.SubjectIndexFilter(
        subject_index,
        exclude={"http://www.yso.fi/onto/yso/p10849"},
    )
    sset = SubjectSet.from_uri_references(
        ["http://www.yso.fi/onto/yso/p10849"], subject_filter
    )
    assert not sset.is_scorable
    assert sset.excluded_annotations == [
        ("http://www.yso.fi/onto/yso/p10849", "excluded_subject")
    ]


def test_subjectset_evidence_unknown_label(subject_index):
    sset = SubjectSet.from_string("nonexistent label\n", subject_index, "fi")
    assert len(sset) == 0
    assert sset.annotation_count == 1
    assert not sset.is_scorable
    assert sset.excluded_annotations == [("nonexistent label", "unknown_label")]


def test_subjectset_evidence_subject_dicts(subject_index):
    sset = SubjectSet.from_subject_dicts(
        [
            {"uri": "http://www.yso.fi/onto/yso/p10849"},
            {"uri": "http://example.org/unknown"},
            {"label": "arkeologit"},
        ],
        subject_index,
        "fi",
    )
    assert len(sset) == 1  # duplicate accepted URI/label deduplicated
    assert sset.annotation_count == 3
    assert sset.excluded_annotations == [
        ("http://example.org/unknown", "unknown_uri")
    ]


def test_subjectset_default_has_no_annotations():
    sset = SubjectSet()
    assert sset.annotation_count == 0
    assert not sset.has_annotations
    assert not sset.is_scorable
    assert sset.excluded_annotations == []


def test_docfile_tsv_evidence(tmpdir, subject_index):
    docfile = tmpdir.join("documents.tsv")
    docfile.write(
        "doc1\t<http://www.yso.fi/onto/yso/p10849>\n"
        "doc2\t<http://example.org/unknown>\n"
        "doc3\t\n"
    )
    docs = list(annif.corpus.DocumentFileTSV(str(docfile), subject_index).documents)
    assert len(docs) == 3
    assert docs[0].subject_set.is_scorable
    assert docs[0].subject_set.annotation_count == 1
    assert not docs[1].subject_set.is_scorable
    assert docs[1].subject_set.has_annotations
    assert docs[1].subject_set.excluded_annotations == [
        ("http://example.org/unknown", "unknown_uri")
    ]
    assert not docs[2].subject_set.has_annotations


def test_docfile_csv_evidence(tmpdir, subject_index):
    docfile = tmpdir.join("documents.csv")
    docfile.write(
        "text,subject_uris\n"
        "doc1,<http://www.yso.fi/onto/yso/p10849>\n"
        "doc2,<http://example.org/unknown>\n"
        "doc3,\n"
    )
    docs = list(annif.corpus.DocumentFileCSV(str(docfile), subject_index).documents)
    assert len(docs) == 3
    assert docs[0].subject_set.is_scorable
    assert not docs[1].subject_set.is_scorable
    assert docs[1].subject_set.excluded_annotations == [
        ("http://example.org/unknown", "unknown_uri")
    ]
    assert not docs[2].subject_set.has_annotations


def test_docfile_jsonl_evidence(tmpdir, subject_index):
    docfile = tmpdir.join("documents.jsonl")
    docfile.write(
        '{"text": "doc1", "subjects": '
        '[{"uri": "http://www.yso.fi/onto/yso/p10849"}]}\n'
        '{"text": "doc2", "subjects": [{"uri": "http://example.org/unknown"}]}\n'
        '{"text": "doc3"}\n'
    )

    # default behavior: subjectless documents are dropped while parsing
    docs = list(
        annif.corpus.DocumentFileJSONL(str(docfile), subject_index, "fi").documents
    )
    assert len(docs) == 1
    assert docs[0].text == "doc1"

    # evaluation behavior: all parseable documents are retained with evidence
    docs = list(
        annif.corpus.DocumentFileJSONL(
            str(docfile), subject_index, "fi", retain_subjectless=True
        ).documents
    )
    assert len(docs) == 3
    assert docs[0].subject_set.is_scorable
    assert not docs[1].subject_set.is_scorable
    assert docs[1].subject_set.has_annotations
    assert not docs[2].subject_set.has_annotations


def test_docdir_retain_subjectless(tmpdir, subject_index):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("<http://www.yso.fi/onto/yso/p10849>")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("<http://example.org/unknown>")
    tmpdir.join("doc3.txt").write("doc3")

    docdir = annif.corpus.DocumentDirectory(
        str(tmpdir),
        subject_index,
        "fi",
        require_subjects=True,
        retain_subjectless=True,
    )
    docs = list(docdir.documents)
    assert len(docs) == 3
    assert docs[0].subject_set.is_scorable
    assert not docs[1].subject_set.is_scorable
    assert docs[1].subject_set.has_annotations
    assert not docs[2].subject_set.has_annotations


def _coverage_doc(subject_index, doc_id, uris):
    return Document(
        text=doc_id,
        subject_set=SubjectSet.from_uri_references(uris, subject_index),
        document_id=doc_id,
    )


def test_gold_coverage(subject_index):
    docs = annif.corpus.DocumentList(
        [
            _coverage_doc(
                subject_index,
                "d1",
                ["http://www.yso.fi/onto/yso/p10849"],
            ),
            _coverage_doc(
                subject_index,
                "d2",
                [
                    "http://www.yso.fi/onto/yso/p1747",
                    "http://example.org/unknown",
                ],
            ),
            _coverage_doc(
                subject_index, "d3", ["http://example.org/unknown2"]
            ),
            Document(text="d4", subject_set=SubjectSet(), document_id="d4"),
        ]
    )
    coverage = GoldCorpusCoverage.from_corpus(docs)
    assert coverage.documents_seen == 4
    assert coverage.scored_documents == 2
    assert coverage.excluded_documents == 2
    assert coverage.unscorable_documents == 1
    assert coverage.without_annotation_documents == 1
    assert coverage.annotations_total == 4
    assert coverage.accepted_annotations == 2
    assert coverage.excluded_annotations == 2
    assert coverage.excluded_annotation_reasons["unknown_uri"] == 2
    records = {rec["document_id"]: rec for rec in coverage.excluded_document_dicts()}
    assert records["d3"]["class"] == "unscorable_annotations"
    assert records["d4"]["class"] == "no_annotations"
    coverage.require_scorable_documents()  # does not raise


def test_gold_coverage_require_scorable_fails(subject_index):
    docs = annif.corpus.DocumentList(
        [
            _coverage_doc(
                subject_index, "d1", ["http://example.org/unknown"]
            ),
            Document(text="d2", subject_set=SubjectSet(), document_id="d2"),
        ]
    )
    coverage = GoldCorpusCoverage.from_corpus(docs)
    with pytest.raises(NotSupportedException) as excinfo:
        coverage.require_scorable_documents()
    assert "no documents with a scorable gold standard" in str(excinfo.value)


def test_gold_coverage_empty_corpus():
    coverage = GoldCorpusCoverage.from_corpus(annif.corpus.DocumentList([]))
    with pytest.raises(NotSupportedException) as excinfo:
        coverage.require_scorable_documents()
    assert "cannot evaluate empty corpus" in str(excinfo.value)


def test_gold_coverage_as_dict(subject_index):
    docs = annif.corpus.DocumentList(
        [
            _coverage_doc(
                subject_index, "d1", ["http://www.yso.fi/onto/yso/p10849"]
            ),
            _coverage_doc(
                subject_index, "d2", ["http://example.org/unknown"]
            ),
        ]
    )
    coverage = GoldCorpusCoverage.from_corpus(docs)
    data = coverage.as_dict()
    assert data["documents_excluded"] == 1
    assert data["documents_with_unscorable_annotations"] == 1
    assert data["documents_without_annotations"] == 0
    assert data["annotations_total"] == 2
    assert data["annotations_accepted"] == 1
    assert data["annotations_excluded"] == 1
    assert data["excluded_annotations_by_reason"]["unknown_uri"] == 1
    assert [rec["document_id"] for rec in data["excluded_documents"]] == ["d2"]


def test_scorable_document_corpus(subject_index):
    docs = annif.corpus.DocumentList(
        [
            _coverage_doc(
                subject_index, "d1", ["http://www.yso.fi/onto/yso/p10849"]
            ),
            _coverage_doc(
                subject_index, "d2", ["http://example.org/unknown"]
            ),
            Document(text="d3", subject_set=SubjectSet(), document_id="d3"),
        ]
    )
    scorable = ScorableDocumentCorpus(docs)
    scorable_docs = list(scorable.documents)
    assert [doc.document_id for doc in scorable_docs] == ["d1"]
    # batches contain scorable documents only, in source order
    scorable.DOC_BATCH_SIZE = 1
    batches = list(scorable.doc_batches)
    assert len(batches) == 1
    assert batches[0][0].document_id == "d1"


def test_cross_format_coverage_consistency(tmpdir, subject_index):
    tsv = tmpdir.join("documents.tsv")
    tsv.write(
        "doc1\t<http://www.yso.fi/onto/yso/p10849>\n"
        "doc2\t<http://example.org/unknown>\n"
        "doc3\t\n"
    )
    csvfile = tmpdir.join("documents.csv")
    csvfile.write(
        "document_id,text,subject_uris\n"
        "d1,doc1,<http://www.yso.fi/onto/yso/p10849>\n"
        "d2,doc2,<http://example.org/unknown>\n"
        'd3,doc3,""\n'
    )
    jsonl = tmpdir.join("documents.jsonl")
    jsonl.write(
        '{"document_id": "d1", "text": "doc1", "subjects": '
        '[{"uri": "http://www.yso.fi/onto/yso/p10849"}]}\n'
        '{"document_id": "d2", "text": "doc2", "subjects": '
        '[{"uri": "http://example.org/unknown"}]}\n'
        '{"document_id": "d3", "text": "doc3"}\n'
    )
    docdir = tmpdir.mkdir("fulltext")
    docdir.join("doc1.txt").write("doc1")
    docdir.join("doc1.key").write("<http://www.yso.fi/onto/yso/p10849>")
    docdir.join("doc2.txt").write("doc2")
    docdir.join("doc2.key").write("<http://example.org/unknown>")
    docdir.join("doc3.txt").write("doc3")

    corpora = [
        annif.corpus.DocumentFileTSV(str(tsv), subject_index),
        annif.corpus.DocumentFileCSV(str(csvfile), subject_index),
        annif.corpus.DocumentFileJSONL(
            str(jsonl), subject_index, "fi", retain_subjectless=True
        ),
        annif.corpus.DocumentDirectory(
            str(docdir),
            subject_index,
            "fi",
            require_subjects=True,
            retain_subjectless=True,
        ),
    ]
    coverages = [GoldCorpusCoverage.from_corpus(corpus) for corpus in corpora]
    for coverage in coverages:
        assert coverage.scored_documents == 1
        assert coverage.excluded_documents == 2
        assert coverage.unscorable_documents == 1
        assert coverage.without_annotation_documents == 1
        assert coverage.annotations_total == 2
        assert coverage.excluded_annotations == 1
    # the scored document set and its order is the same across formats
    scored_ids = [
        [doc.text for doc in ScorableDocumentCorpus(corpus).documents]
        for corpus in corpora
    ]
    assert scored_ids == [["doc1"], ["doc1"], ["doc1"], ["doc1"]]


def test_limitingcorpus(tmpdir, subject_index):
    docfile = tmpdir.join("documents_invalid.tsv")
    docfile.write(
        """Läntinen\t<http://www.yso.fi/onto/yso/p2557>

        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        A line with no tabs
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>"""
    )

    document_corpus = annif.corpus.DocumentFileTSV(str(docfile), subject_index)
    limiting_corpus = annif.corpus.LimitingDocumentCorpus(document_corpus, 2)

    assert len(list(limiting_corpus.documents)) == 2
    for limited_doc, doc in zip(limiting_corpus.documents, document_corpus.documents):
        assert limited_doc.text == doc.text
