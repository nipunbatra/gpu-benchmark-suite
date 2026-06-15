FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt update && apt install -y git wget unzip libgl1 libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ultralytics pulls in an opencv-python whose cv2.dnn lacks DictValue (breaks the
# detection benchmark). Force the headless build, which is correct for a server anyway.
RUN pip uninstall -y opencv-python opencv-python-headless || true && \
    pip install --no-cache-dir "opencv-python-headless==4.8.1.78"
