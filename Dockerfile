# Multi-stage build for optimal image size
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Install runtime dependencies (cron for scheduled tasks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    curl \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application files
COPY src/ ./src/
COPY requirements.txt .
COPY *.py .

# Create necessary directories
RUN mkdir -p /app/logs /app/models

# Copy Docker-specific files first
COPY docker-entrypoint.sh /docker-entrypoint.sh
COPY docker-crontab /etc/cron.d/fusionforecast

# Set permissions and install crontab
RUN dos2unix /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh && \
    dos2unix /etc/cron.d/fusionforecast && \
    chmod 0644 /etc/cron.d/fusionforecast && \
    touch /var/log/cron.log && \
    crontab /etc/cron.d/fusionforecast

# Health check
HEALTHCHECK --interval=5m --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "from src.config import settings; print('OK')" || exit 1

# Entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["cron", "-f"]
