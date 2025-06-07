# Multi-stage build for smaller final image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /build

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 blackwall

# Copy Python packages from builder
COPY --from=builder /root/.local /home/blackwall/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY README.md .

# Install application
RUN pip install --no-cache-dir -e .

# Create directories for models and cache
RUN mkdir -p /app/models /app/cache && \
    chown -R blackwall:blackwall /app

# Switch to non-root user
USER blackwall

# Add local bin to PATH
ENV PATH="/home/blackwall/.local/bin:${PATH}"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Download models (in production, these would be included or mounted)
RUN blackwall --help

ENTRYPOINT ["blackwall"]