# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/tensorrt:23.08-py3

ARG DITTO_COMMIT=c3e47eee2e626500017a0556b470d6d4182f85e8

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ffmpeg git libegl1 libgl1 libgles2 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt

RUN python -m pip install \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && python -m pip install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

RUN mkdir -p /opt/ditto-talkinghead \
    && cd /opt/ditto-talkinghead \
    && git init \
    && git remote add origin https://github.com/antgroup/ditto-talkinghead.git \
    && git fetch --depth 1 origin "${DITTO_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && sed -i 's/np\.atan2/np.arctan2/g' core/aux_models/mediapipe_landmark478.py \
    && grep -q 'np.arctan2' core/aux_models/mediapipe_landmark478.py \
    && rm -rf .git

WORKDIR /opt/livetalking
COPY . .
COPY data/avatars/ditto_woman /opt/default-avatars/ditto_woman
COPY data/avatars/ditto_man /opt/default-avatars/ditto_man
COPY data/avatars/ditto_man_clinic /opt/default-avatars/ditto_man_clinic
COPY data/avatars/ditto_woman_teacher /opt/default-avatars/ditto_woman_teacher
COPY data/avatars/ditto_malaywoman /opt/default-avatars/ditto_malaywoman
COPY data/avatars/ditto_caucasian /opt/default-avatars/ditto_caucasian

RUN rm -rf data \
    && chmod +x docker/start.sh \
    && python -m compileall -q app.py avatars server streamout tts utils \
    && python - <<'PY'
import onnxruntime as ort
import tensorrt as trt

assert trt.__version__.startswith("8.6.1"), trt.__version__
assert "CUDAExecutionProvider" in ort.get_available_providers(), ort.get_available_providers()
print("TensorRT", trt.__version__)
print("ONNX Runtime", ort.__version__, ort.get_available_providers())
PY

# Stamped by CI so the running container can prove which commit it is.
ARG BUILD_SHA=dev
ENV BUILD_SHA=${BUILD_SHA}

EXPOSE 8010 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD curl -fsS http://127.0.0.1:${LISTEN_PORT:-8010}/index-en.html >/dev/null \
        || curl -fsS http://127.0.0.1:8888/ >/dev/null \
        || exit 1

ENTRYPOINT ["/opt/livetalking/docker/start.sh"]
