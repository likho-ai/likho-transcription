# syntax=docker/dockerfile:1
# CPU image. The speech model is not baked in: it is downloaded on first use into /models,
# which should be a volume so the download happens once.
FROM python:3.12-slim AS build
# git: the contracts and hinglish packages are installed from their repositories at a tag.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.12.17
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
RUN useradd --system --uid 10001 --no-create-home likho \
    && mkdir /models && chown likho /models
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 HF_HOME=/models
USER likho
VOLUME /models
EXPOSE 5020 4020
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:4020/readyz', timeout=2).status == 200 else 1)"]
CMD ["likho-transcription"]
