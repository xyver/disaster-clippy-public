FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port (Railway sets $PORT)
EXPOSE 8000

# Start command - Railway sets PORT env var
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
