FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt update && apt install -y python3 python3-pip git wget unzip libgl1 libglib2.0-0
RUN ln -s /usr/bin/python3 /usr/bin/python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
