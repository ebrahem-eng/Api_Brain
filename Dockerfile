FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Install numpy first
RUN pip install --no-cache-dir numpy==1.23.5

# Install torch CPU build
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install the rest
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    python-multipart==0.0.9 \
    opencv-python-headless==4.8.1.78 \
    ultralytics==8.0.196

COPY . .

EXPOSE 10000

CMD ["python", "app.py"]
