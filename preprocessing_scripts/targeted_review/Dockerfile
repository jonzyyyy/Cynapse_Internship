# Start from an official Python image
FROM python:3.12-slim

# Set a working directory
WORKDIR /app

# Install system dependencies (add more if your scripts need them, e.g., git, ffmpeg, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy your code
COPY . .

# Set the entrypoint (optional, you can override with docker run)
CMD ["python3", "pipeline.py"]