FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN apt-get update
RUN apt-get install libgl1-mesa-glx  -y
RUN pip install --no-cache-dir ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
RUN pip install --no-cache-dir openvino-dev
RUN pip install --no-cache-dir opencv-python-headless==4.4.0.44 opencv-python==4.4.0.44
RUN pip install --no-cache-dir onnxruntime-gpu
RUN pip install pycocotools numpy tqdm pyyaml webcolors matplotlib

WORKDIR /detector