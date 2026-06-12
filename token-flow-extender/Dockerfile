FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE model_limits.yaml ./
COPY src ./src
COPY examples ./examples

RUN pip install --no-cache-dir ".[fastapi]"

EXPOSE 8800

CMD ["uvicorn", "examples.fastapi_integration:app", "--host", "0.0.0.0", "--port", "8800"]
