# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /app

# Build dependencies (wheels are preferred, but this keeps builds reliable)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source
COPY pyproject.toml README.md LICENSE ./
COPY hawkins_truth_engine ./hawkins_truth_engine

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .


FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Runtime libs used by lxml
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Keep the source available (editable install points here)
COPY pyproject.toml README.md LICENSE ./
COPY hawkins_truth_engine ./hawkins_truth_engine

EXPOSE 8000

CMD ["python", "-m", "hawkins_truth_engine.app", "--host", "0.0.0.0", "--port", "8000"]
