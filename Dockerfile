# ==== Base image: TF 2.5.1 with Jupyter (works on M1 with emulation) ====
FROM --platform=linux/amd64 tensorflow/tensorflow:2.5.1-jupyter

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ==== System packages needed for Object Detection API ====
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        protobuf-compiler \
        python3-dev \
        build-essential \
        wget \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# ==== Python dependencies ====
RUN pip install --no-cache-dir \
    pillow \
    lxml \
    Cython \
    contextlib2 \
    matplotlib \
    pandas \
    opencv-python-headless \
    pycocotools \
    tf_slim \
    protobuf==3.19.4

# ==== Clone TF Models Repo (working OD API version) ====
RUN git clone https://github.com/tensorflow/models /models && \
    cd /models && git checkout 8d9ce6a58e5ef25e7bae647fdfb77b4b5c5d42e1

# ==== Build & Install Object Detection API ====
WORKDIR /models/research

# Compile protobuf files
RUN protoc object_detection/protos/*.proto --python_out=.

# Install COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    python3 setup.py build_ext --inplace && \
    python3 setup.py install

# Install Object Detection API as a package
RUN cp object_detection/packages/tf2/setup.py . && \
    python3 -m pip install --no-cache-dir .

# Back to notebook working directory
WORKDIR /tf

