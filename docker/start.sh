#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=/opt/livetalking
DITTO_ROOT=/opt/ditto-talkinghead
WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace}
CACHE_ROOT=${CACHE_ROOT:-$WORKSPACE_ROOT/cache}
AVATAR_ID=${AVATAR_ID:-ditto_man}
DEFAULT_AVATAR_ROOT=/opt/default-avatars

# MODEL_ROOT + every DITTO_*/ASR_* default. Sourced (not duplicated) so a shell
# in JupyterLab can reproduce the server's exact rendering parameters — see the
# header of docker/ditto-env.sh.
# shellcheck source=docker/ditto-env.sh
source "$APP_ROOT/docker/ditto-env.sh"

if compgen -G "$WORKSPACE_ROOT/LiveTalking/data/avatars/$AVATAR_ID/source.*" >/dev/null; then
    DATA_ROOT="$WORKSPACE_ROOT/LiveTalking/data"
else
    DATA_ROOT=${DATA_ROOT:-$WORKSPACE_ROOT/data}
fi

# Jupyter comes up before any /workspace I/O: copying avatars onto a cold
# network volume can take minutes, and that is exactly when we need a shell.
JUPYTER_TOKEN=${JUPYTER_TOKEN:-$(python -c 'import secrets; print(secrets.token_urlsafe(24))')}
echo "=== livetalking-avatar build ${BUILD_SHA:-unknown} ==="
echo "JupyterLab: http://<pod-host>:8888/?token=$JUPYTER_TOKEN"
jupyter lab \
    --allow-root \
    --no-browser \
    --ip=0.0.0.0 \
    --port=8888 \
    --ServerApp.root_dir="$WORKSPACE_ROOT" \
    --ServerApp.allow_remote_access=True \
    --IdentityProvider.token="$JUPYTER_TOKEN" &

mkdir -p "$DATA_ROOT/avatars" "$MODEL_ROOT" "$CACHE_ROOT/modelscope" "$CACHE_ROOT/huggingface"
for bundled_avatar in "$DEFAULT_AVATAR_ROOT"/*; do
    [[ -d "$bundled_avatar" ]] || continue
    bundled_id=$(basename "$bundled_avatar")
    if ! compgen -G "$DATA_ROOT/avatars/$bundled_id/source.*" >/dev/null; then
        cp -a "$bundled_avatar" "$DATA_ROOT/avatars/$bundled_id"
        echo "Installed bundled avatar: $bundled_id"
    fi
done
ln -sfn "$DATA_ROOT" "$APP_ROOT/data"

export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}
export HF_HOME=${HF_HOME:-$CACHE_ROOT/huggingface}
export PYTHONPATH="$DITTO_ROOT:$APP_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PY_SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
for lib_dir in nvidia/cuda_runtime/lib nvidia/cublas/lib nvidia/cudnn/lib nvidia/cufft/lib; do
    if [[ -d "$PY_SITE/$lib_dir" ]]; then
        export LD_LIBRARY_PATH="$PY_SITE/$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done

if [[ -z "${ELEVENLABS_API_KEY:-}" ]]; then
    echo "ERROR: Set ELEVENLABS_API_KEY in the RunPod template." >&2
    exit 2
fi

while ! compgen -G "$DATA_ROOT/avatars/$AVATAR_ID/source.*" >/dev/null; do
    echo "Waiting for $DATA_ROOT/avatars/$AVATAR_ID/source.mp4 (upload it through JupyterLab)..."
    sleep 5
done

if [[ ! -f "$MODEL_ROOT/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" || \
      ! -f "$MODEL_ROOT/ditto_trt_Ampere_Plus/warp_network_fp16.engine" ]]; then
    echo "Ditto checkpoints are missing; downloading them once to $MODEL_ROOT ..."
    hf download digital-avatar/ditto-talkinghead \
        --include "ditto_cfg/*" "ditto_trt_Ampere_Plus/*" \
        --local-dir "$MODEL_ROOT"
fi

if [[ "$DITTO_GENERATE_IDLE" == "1" ]]; then
    echo "Checking decoder-matched idle for $AVATAR_ID ..."
    if ! python "$APP_ROOT/scripts/ditto_make_idle.py" \
        --avatar "$AVATAR_ID" \
        --data-root "$DATA_ROOT/avatars" \
        --out "$DATA_ROOT/avatars/$AVATAR_ID/idle.generated.mp4" \
        --if-stale; then
        echo "WARNING: generated idle failed; using original idle.mp4" >&2
        rm -f "$DATA_ROOT/avatars/$AVATAR_ID/idle.generated.mp4.tmp.mp4" \
              "$DATA_ROOT/avatars/$AVATAR_ID/idle.generated.mp4.json.tmp" \
              "$DATA_ROOT/avatars/$AVATAR_ID/idle.generated.mp4.json"
    fi
fi

cd "$APP_ROOT"
exec python app.py \
    --model ditto \
    --avatar_id "$AVATAR_ID" \
    --transport webrtc \
    --tts elevenlabs \
    --REF_FILE "${VOICE_ID:-aSXZu6bgEOS8MXVRzjPi}" \
    --listenport "${LISTEN_PORT:-8010}" \
    --fps 25 \
    "$@"
