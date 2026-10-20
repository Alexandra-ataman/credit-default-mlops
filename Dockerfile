FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GX_ANALYTICS_ENABLED=false
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir '.[api]'

COPY models/model.joblib ./models/model.joblib
RUN useradd --create-home app
USER app
EXPOSE 8000
CMD ["uvicorn", "credit_default.api:app", "--host", "0.0.0.0", "--port", "8000"]
