# Update Dockerfile base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04  # More compatible base
RUN apt-get update && apt-get install -y \
    python3.10 \
    ffmpeg \
    python3-pip \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt && \
    python -m playwright install chromium && \
    playwright install-deps

# Download VOSK model
RUN mkdir -p models \
    && wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -O models/vosk.zip \
    && unzip models/vosk.zip -d models/ \
    && mv models/vosk-model-small-en-us-0.15 models/vosk-model-small-en-us \
    && rm models/vosk.zip

# Start Ollama and JENNA
CMD ollama serve & \
    sleep 10 && \
    python services/main_service.py
    
COPY ./app /app
WORKDIR /app
CMD ["python", "-m", "services.main_service"]