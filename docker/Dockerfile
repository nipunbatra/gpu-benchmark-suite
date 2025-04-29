FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip git wget unzip
RUN ln -s /usr/bin/python3 /usr/bin/python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt