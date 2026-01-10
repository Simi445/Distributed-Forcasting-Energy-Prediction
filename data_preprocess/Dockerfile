FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    ca-certificates \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-ca-certificates

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8001"]