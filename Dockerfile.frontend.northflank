# Frontend only. Build after recording; never publish the original runtime/ folder.
ARG PYTHON_IMAGE=python:3.13.18-slim-bookworm
# Optional export target: inspect the context using BuildKit's real ignore rules.
FROM scratch AS northflank-context
COPY . /

FROM ${PYTHON_IMAGE} AS frontend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/app \
    APP_ENV=northflank \
    NORTHFLANK_PUBLIC_READ_ONLY=true \
    NORTHFLANK_PUBLIC_DATA_APPROVED=false \
    STORAGE_PROVIDER=supabase \
    PRESCRIPTIVE_NEWS_ENABLED=false \
    ACADEMIC_DEMO_ENABLED=false \
    DASHBOARD_FALLBACK_ENABLED=false \
    MODEL_CONFIDENCE_THRESHOLD=0.60

WORKDIR /app
COPY deploy/northflank/requirements.frontend.txt /app/deploy/northflank/requirements.frontend.txt
RUN pip install --no-cache-dir --only-binary=:all: -r /app/deploy/northflank/requirements.frontend.txt

RUN useradd --uid 10001 --user-group --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/.streamlit
COPY --chown=app:app alpha_engine/ /app/alpha_engine/
COPY --chown=app:app models/xgboost_v1.joblib models/xgboost_v1.sha256 models/xgboost_v1_metadata.json /app/models/
COPY --chown=app:app models/baseline_v1.json models/economic_summary_v1.json /app/models/
COPY --chown=app:app deploy/northflank/streamlit.toml /app/.streamlit/config.toml
COPY --chown=app:app deploy/northflank/entrypoint.py /app/deploy/northflank/entrypoint.py
COPY --chown=app:app deploy/northflank/public_config.py deploy/northflank/public_data.py deploy/northflank/public_views.py deploy/northflank/public_app.py /app/deploy/northflank/
COPY --chown=app:app deploy/northflank/public_presentation.py deploy/northflank/public_replay.py /app/deploy/northflank/
COPY --chown=app:app deploy/northflank/academic_client.py deploy/northflank/academic_panel.py /app/deploy/northflank/

# Verify the existing artifact; do not train or load it for inference.
RUN python -c "import hashlib,pathlib; p=pathlib.Path('/app/models/xgboost_v1.joblib'); expected=p.with_suffix('.sha256').read_text().strip(); assert hashlib.sha256(p.read_bytes()).hexdigest()==expected, 'V1 checksum mismatch'"

USER app:app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3); assert r.status==200"
ENTRYPOINT ["python", "/app/deploy/northflank/entrypoint.py"]
