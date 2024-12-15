# Base image
FROM python:3.10.3-slim-bullseye

# Install system dependencies
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libatlas-base-dev \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    python3-numpy \
    curl \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Copy .env file
COPY .env .env

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 4000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "4000"]
