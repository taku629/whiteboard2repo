FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt coverage
COPY . .
ENV PORT=8000
# MUSE_API_KEY must be provided at runtime (OpenAI-compatible endpoint serving Muse Spark Open)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
