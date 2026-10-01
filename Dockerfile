FROM python:3.11.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# main2.py, templates/index.html and load_data/ must sit next to this Dockerfile
COPY . .

# one worker on purpose: each user's data lives in this process's memory
CMD ["sh", "-c", "uvicorn main2:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]