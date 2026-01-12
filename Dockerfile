FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    cmake \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Сборка llama.cpp сервера
WORKDIR /tmp
RUN git clone https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build -DGGML_CUDA=ON && \
    cmake --build build --config Release --target llama-server -j $(nproc) && \
    mv build/bin/llama-server /usr/local/bin/llama-server && \
    cd / && \
    rm -rf /tmp/llama.cpp

WORKDIR /app

COPY requirements.txt .
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py && \
    pip3.12 install --no-cache-dir --upgrade pip && \
    pip3.12 install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/
COPY templates/ ./templates/

COPY data/model/ ./data/model/

EXPOSE 8000

CMD ["python3.12", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
