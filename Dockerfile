FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY synthlens ./synthlens
RUN pip install --no-cache-dir .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "synthlens.server.app:app", "--host", "0.0.0.0", "--port", "8000"]