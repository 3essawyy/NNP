# Reproducible inference environment (brief §5.4, ADR-009).
#
#   docker build -t nnp-har .
#   docker run --rm -v /path/to/data:/data:ro -v "$PWD/out":/out nnp-har
#
# results.txt and time.txt are written to /out (the container's working directory).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY infer.py ./
COPY src ./src
COPY models/model.pkl ./models/model.pkl

# Fail the build if the model cannot be loaded (catches version drift early).
RUN python -c "import sys; sys.path.insert(0, 'src'); \
from nnp.models.bundle import load_bundle; from pathlib import Path; \
load_bundle(Path('models/model.pkl')); print('model OK')"

RUN useradd --create-home --uid 1000 runner && mkdir /out && chown runner /out
USER runner
WORKDIR /out
ENTRYPOINT ["python", "/app/infer.py"]
CMD ["/data"]
