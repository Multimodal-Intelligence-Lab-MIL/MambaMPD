# MambaMPD demo image.
#
# NOTE: the MambaMPD encoder relies on the `mamba-ssm` CUDA kernels, which
# require an NVIDIA GPU and a CUDA toolchain to build. For a full training /
# inference image use an `nvidia/cuda` base; this slim image is intended for
# the lightweight Streamlit demo when CPU-compatible builds of the
# dependencies are available.

FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./models ./models
COPY ./utils ./utils
COPY ./app.py .
COPY ./sample_padding_image_for_inference/img_0814.jpg /data/images/img_0814.jpg

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.port=8000"]
