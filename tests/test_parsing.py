import pytest

from backend.parsing import ResumeFileError, extract_text, validate_file
from tests.conftest import SAMPLES


def test_docx_text_and_links():
    text = extract_text((SAMPLES / "sample_resume.docx").read_bytes(), "sample_resume.docx")
    assert "Expense Tracker API" in text and "EDUCATION" in text


def test_pdf_text_and_hyperlink():
    text = extract_text((SAMPLES / "sample_resume.pdf").read_bytes(), "sample_resume.pdf")
    assert "Sample Candidate" in text
    assert "github.com/sample-candidate" in text          # link annotation is appended


@pytest.mark.parametrize("data,name", [
    (b"", "a.pdf"), (b"hello", "a.pdf"), (b"hello", "a.docx"), (b"x", "a.txt"), (b"x", "a.doc"),
])
def test_bad_files_are_rejected(data, name):
    with pytest.raises(ResumeFileError):
        validate_file(data, name)


def test_too_big(monkeypatch):
    monkeypatch.setattr("backend.config.MAX_FILE_MB", 0.0001)
    with pytest.raises(ResumeFileError, match="larger"):
        validate_file(b"%PDF" + b"0" * 500, "a.pdf")
