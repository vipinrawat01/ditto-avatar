"""Generate a decoder-matched idle clip without replacing the original."""

import argparse
import hashlib
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avatars.ditto_avatar import (  # noqa: E402
    neutralize_sdk_source_lips,
    setup_kwargs_from_env,
)

CHUNKSIZE = (3, 5, 2)
SPLIT_LEN = int(sum(CHUNKSIZE) * 0.04 * 16000) + 80
FPS = 25
GENERATOR_VERSION = 3


class Collector:
    """Collect frames from Ditto's single writer thread in output order."""

    def __init__(self):
        self.frames = []
        self.last = None
        self.lock = threading.Lock()

    def __call__(self, frame_rgb, fmt="rgb"):
        with self.lock:
            self.frames.append(np.asarray(frame_rgb))
            self.last = time.perf_counter()

    def close(self):
        pass


def ping_pong(frames):
    """Mirror a sequence without repeating either endpoint."""
    return frames + frames[-2:0:-1] if len(frames) > 2 else frames


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fingerprint(source, idle, kwargs, seconds, ping_pong_enabled):
    payload = {
        "version": GENERATOR_VERSION,
        "source_sha256": file_sha256(source),
        "idle_sha256": file_sha256(idle),
        "kwargs": kwargs,
        "seconds": seconds,
        "ping_pong": ping_pong_enabled,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avatar", default=os.environ.get("AVATAR_ID", "ditto_woman"))
    parser.add_argument(
        "--data-root", default=os.environ.get("DITTO_AVATAR_DATA", "data/avatars"))
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--ditto-repo", default=os.environ.get("DITTO_REPO", "/opt/ditto-talkinghead"))
    parser.add_argument(
        "--cfg", default=os.environ.get("DITTO_CFG"))
    parser.add_argument(
        "--sdk-data-root", default=os.environ.get("DITTO_DATA_ROOT"))
    parser.add_argument("--no-ping-pong", action="store_true")
    parser.add_argument("--if-stale", action="store_true")
    args = parser.parse_args()

    if not args.cfg or not args.sdk_data_root:
        sys.exit("set DITTO_CFG and DITTO_DATA_ROOT")

    avatar_dir = os.path.join(args.data_root, args.avatar)
    sources = [
        os.path.join(avatar_dir, name)
        for name in sorted(os.listdir(avatar_dir))
        if name.startswith("source.")
    ]
    if not sources:
        sys.exit(f"no source.* in {avatar_dir}")

    source = sources[0]
    original_idle = os.path.join(avatar_dir, "idle.mp4")
    if not os.path.exists(original_idle):
        sys.exit(f"no original idle.mp4 in {avatar_dir}")
    out = args.out or os.path.join(avatar_dir, "idle.generated.mp4")

    kwargs = setup_kwargs_from_env()
    fingerprint, fingerprint_data = build_fingerprint(
        source, original_idle, kwargs, args.seconds, not args.no_ping_pong)
    metadata_path = out + ".json"
    if args.if_stale and os.path.exists(out) and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                if json.load(handle).get("fingerprint") == fingerprint:
                    print(f"generated idle is current: {out}")
                    return
        except (OSError, ValueError):
            pass
    elif os.path.exists(out):
        sys.exit(f"{out} exists; pass --if-stale to replace it safely")

    sys.path.insert(0, args.ditto_repo)
    from stream_pipeline_online import StreamSDK

    print(f"speech_source={source}\nidle_source={original_idle}\nkwargs={kwargs}")
    sdk = StreamSDK(args.cfg, args.sdk_data_root)
    # Use the recorded closed-mouth idle as this isolated SDK's source. Using
    # the speaking source here leaks its mouth motion into silence and makes the
    # generated clip look like source.mp4 playback before speech begins.
    sdk.setup(original_idle, "/tmp/ditto_make_idle_dummy.mp4", **kwargs)
    neutralize_sdk_source_lips(sdk, original_idle)

    collector = Collector()
    sdk.writer = collector
    wanted = int(args.seconds * FPS)
    runs = (wanted + CHUNKSIZE[1] - 1) // CHUNKSIZE[1]
    silence = np.zeros(SPLIT_LEN, dtype=np.float32)
    for _ in range(runs):
        sdk.run_chunk(silence, CHUNKSIZE)

    deadline = time.perf_counter() + max(120.0, args.seconds * 20)
    while time.perf_counter() < deadline:
        with collector.lock:
            count, last = len(collector.frames), collector.last
        if count >= runs * CHUNKSIZE[1]:
            break
        if last and time.perf_counter() - last > 15:
            print(f"SDK stopped producing at {count} frames")
            break
        time.sleep(0.2)

    with collector.lock:
        frames = list(collector.frames)
    frames = frames[CHUNKSIZE[1] * 2:]
    if len(frames) < FPS:
        sys.exit(f"only {len(frames)} usable frames; check cfg, models and GPU")
    if not args.no_ping_pong:
        frames = ping_pong(frames)

    height, width = frames[0].shape[:2]
    temporary_out = out + ".tmp.mp4"
    writer = cv2.VideoWriter(
        temporary_out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        sys.exit(f"cannot open {temporary_out} for writing")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    os.replace(temporary_out, out)

    metadata = {"fingerprint": fingerprint, **fingerprint_data}
    temporary_metadata = metadata_path + ".tmp"
    with open(temporary_metadata, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    os.replace(temporary_metadata, metadata_path)

    try:
        sdk.close()
    except Exception:
        pass
    print(
        f"wrote {out}: {len(frames)} frames @ {FPS}fps, "
        f"{width}x{height} ({len(frames) / FPS:.1f}s loop)")


def selfcheck():
    assert ping_pong([1, 2, 3, 4]) == [1, 2, 3, 4, 3, 2]
    assert ping_pong([1, 2]) == [1, 2]
    assert ping_pong([1]) == [1]
    print("OK")


if __name__ == "__main__":
    selfcheck() if "--self-test" in sys.argv else main()
