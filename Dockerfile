# engine: OpenAI-compatible LLM server. Needs the NVIDIA Container Toolkit on the host.
#
#   docker build -t engine .
#   docker run --gpus all -p 8000:8000 -v $PWD/models:/models engine \
#       --model /models/Llama-3.2-1B-Instruct
#
# PyTorch wheels bundle the CUDA runtime, so a slim Python base is enough; the host only
# needs a driver new enough for CUDA 13.2.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Pinned: CUDA 13.2 build of PyTorch 2.15 (brings its matching Triton on Linux).
RUN pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cu132

COPY pyproject.toml README.md ./
COPY engine ./engine
RUN pip install ".[serve]"

RUN useradd --create-home engine
USER engine

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

ENTRYPOINT ["engine", "serve", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--model", "/models/Llama-3.2-1B-Instruct"]
