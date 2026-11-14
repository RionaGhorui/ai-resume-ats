"""Turn an uploaded PDF/DOCX into plain text."""
import io
import re

import pdfplumber
from docx import Document

from backend import config


class ResumeFileError(ValueError):
    """The uploaded file is unusable. The message is safe to show to the user."""


def validate_file(data: bytes, filename: str) -> str:
    """Return the extension ('pdf' or 'docx') or raise ResumeFileError."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "doc":
        raise ResumeFileError("Old .doc files are not supported. Save it as .docx or .pdf.")
    if ext not in ("pdf", "docx"):
        raise ResumeFileError("Only .pdf and .docx files are supported.")
    if not data:
        raise ResumeFileError("The uploaded file is empty.")
    if len(data) > config.MAX_FILE_MB * 1024 * 1024:
        raise ResumeFileError(f"File is larger than {config.MAX_FILE_MB} MB.")
    # check the real file signature instead of trusting the extension
    if ext == "pdf" and not data.startswith(b"%PDF"):
        raise ResumeFileError("This file is not a valid PDF.")
    if ext == "docx" and not data.startswith(b"PK"):
        raise ResumeFileError("This file is not a valid .docx document.")
    return ext


def _pdf_text(data: bytes) -> str:
    parts, links = [], []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
            # links hidden behind words like "GitHub" are invisible in plain text
            links += [h["uri"] for h in page.hyperlinks if h.get("uri")]
    return "\n".join(parts) + ("\n" + "\n".join(dict.fromkeys(links)) if links else "")


def _docx_text(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    lines = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            lines.append(" | ".join(c.text.strip() for c in row.cells))
    links = [r.target_ref for r in doc.part.rels.values() if r.reltype.endswith("/hyperlink")]
    return "\n".join(lines) + ("\n" + "\n".join(dict.fromkeys(links)) if links else "")


def extract_text(data: bytes, filename: str) -> str:
    ext = validate_file(data, filename)
    try:
        text = _pdf_text(data) if ext == "pdf" else _docx_text(data)
    except Exception as exc:  # corrupt or password-protected files
        raise ResumeFileError("Could not read this file. Is it corrupted or password-protected?") from exc
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < config.MIN_TEXT_CHARS:
        raise ResumeFileError(
            "Almost no text found. Scanned/image-only resumes are not supported (no OCR)."
        )
    return text
