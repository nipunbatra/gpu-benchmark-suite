FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt update && apt install -y git wget unzip libgl1 libglib2.0-0

COPY requirements.txt .
RUN pip install uv
RUN uv pip install -r requirements.txt
