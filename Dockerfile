# One container, one URL: Streamlit (public) talks to FastAPI (localhost only) inside it.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    EMBEDDING_CACHE_DIR=/app/.cache/fastembed \
    BACKEND_URL=http://127.0.0.1:8000 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding(os.environ['EMBEDDING_MODEL'], cache_dir=os.environ['EMBEDDING_CACHE_DIR'])"

COPY backend backend
COPY frontend frontend
COPY .streamlit .streamlit
COPY start.sh .

#run as a non-root user
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)"

CMD ["bash", "start.sh"]
