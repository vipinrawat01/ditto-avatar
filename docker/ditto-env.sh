# Every DITTO_* value the server runs with, in one sourceable place.
#
# start.sh sources this, and so must anything that has to match the running
# server -- notably scripts/ditto_make_idle.py, whose whole job is to produce
# frames that match generated speech pixel for pixel:
#
#     cd /opt/livetalking && source docker/ditto-env.sh
#     python scripts/ditto_make_idle.py --avatar ditto_woman
#
# Exported vars in the shell win, so the RunPod template still overrides these.
# Do NOT copy these numbers anywhere else; a second copy will drift.

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace}
DITTO_ROOT=${DITTO_ROOT:-/opt/ditto-talkinghead}

OLD_MODEL_ROOT="$WORKSPACE_ROOT/ditto-talkinghead/checkpoints"
if [[ -n "${DITTO_CHECKPOINTS:-}" ]]; then
    MODEL_ROOT="$DITTO_CHECKPOINTS"
elif [[ -f "$OLD_MODEL_ROOT/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" && \
        -f "$OLD_MODEL_ROOT/ditto_trt_Ampere_Plus/warp_network_fp16.engine" ]]; then
    MODEL_ROOT="$OLD_MODEL_ROOT"
else
    MODEL_ROOT="$WORKSPACE_ROOT/models/ditto"
fi
export MODEL_ROOT

export DITTO_REPO=${DITTO_REPO:-$DITTO_ROOT}
export DITTO_CFG=${DITTO_CFG:-$MODEL_ROOT/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl}
export DITTO_DATA_ROOT=${DITTO_DATA_ROOT:-$MODEL_ROOT/ditto_trt_Ampere_Plus}

# Rendering. These four decide what a generated frame looks like, so idle.mp4
# must be regenerated whenever any of them changes.
export DITTO_STEPS=${DITTO_STEPS:-5}
export DITTO_MAX_SIZE=${DITTO_MAX_SIZE:-1280}
export DITTO_EMO=${DITTO_EMO:-0}
export DITTO_EXP=${DITTO_EXP:-0.63}
export DITTO_ONLINE=${DITTO_ONLINE:-1}
export DITTO_SMO_K_D=${DITTO_SMO_K_D:-1}
export DITTO_LIP_RESPONSE=${DITTO_LIP_RESPONSE:-1.4}
export DITTO_PAUSE_CLOSE_MS=${DITTO_PAUSE_CLOSE_MS:-120}
export DITTO_VAD_RMS=${DITTO_VAD_RMS:-0.004}

# Playback timing. Does not affect how a frame is rendered.
export DITTO_FEED_CAP=${DITTO_FEED_CAP:-20}
export DITTO_START_BUFFER=${DITTO_START_BUFFER:-6}
export DITTO_HOLD=${DITTO_HOLD:-0.10}
export DITTO_TAIL_MS=${DITTO_TAIL_MS:-500}
export DITTO_AV_OFFSET_MS=${DITTO_AV_OFFSET_MS:-220}
# Keep the final transition independent from the A/V calibration offset. Raising
# this value freezes the final generated frame longer before the 160ms blend.
export DITTO_FINAL_HOLD_MS=${DITTO_FINAL_HOLD_MS:-240}

# Keep the recorded closed-mouth idle.mp4 as the visible idle. Generation is
# opt-in, and generating a comparison clip never makes the server play it.
export DITTO_GENERATE_IDLE=${DITTO_GENERATE_IDLE:-0}
export DITTO_USE_GENERATED_IDLE=${DITTO_USE_GENERATED_IDLE:-0}

export ASR_BACKEND=${ASR_BACKEND:-browser}
export ASR_MODEL=${ASR_MODEL:-Qwen/Qwen3-ASR-0.6B}
export ASR_DEVICE=${ASR_DEVICE:-cuda:0}
export ASR_LANGUAGE=${ASR_LANGUAGE:-English}
export CHAT_AGENT_ID=${CHAT_AGENT_ID:-DEVELOPMENT_HPB_17_215536fc419d4d45a6148239df3b1ba8}
