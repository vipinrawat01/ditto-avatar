import os
import sys
import time
import argparse
import threading

import numpy as np


CHUNKSIZE = (3, 5, 2)
SPLIT_LEN = int(sum(CHUNKSIZE) * 0.04 * 16000) + 80


class Sink:
    def __init__(self):
        self.n = 0
        self.t0 = None
        self.last = None
        self.lock = threading.Lock()

    def __call__(self, frame_rgb, fmt="rgb"):
        with self.lock:
            now = time.perf_counter()
            self.t0 = self.t0 or now
            self.last = now
            self.n += 1

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/workspace/LiveTalking/data/avatars/ditto_woman/source.png")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--ditto-repo", default=os.environ.get("DITTO_REPO", "/workspace/ditto-talkinghead"))
    ap.add_argument("--cfg", default=os.environ.get("DITTO_CFG", "/workspace/ditto-talkinghead/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"))
    ap.add_argument("--data-root", default=os.environ.get("DITTO_DATA_ROOT", "/workspace/ditto-talkinghead/checkpoints/ditto_pytorch"))
    ap.add_argument("--steps", type=int, default=int(os.environ.get("DITTO_STEPS", "5")))
    ap.add_argument("--max-size", type=int, default=int(os.environ.get("DITTO_MAX_SIZE", "1024")))
    ap.add_argument("--emo", type=int, default=int(os.environ.get("DITTO_EMO", "4")))
    ap.add_argument("--online", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.ditto_repo)
    from stream_pipeline_online import StreamSDK

    sdk = StreamSDK(args.cfg, args.data_root)
    sdk.setup(
        args.source,
        "/tmp/ditto_bench.mp4",
        sampling_timesteps=args.steps,
        max_size=args.max_size,
        emo=args.emo,
        online_mode=args.online,
    )
    sink = Sink()
    sdk.writer = sink

    total = int(args.seconds * 25)
    runs = (total + CHUNKSIZE[1] - 1) // CHUNKSIZE[1]
    audio = np.zeros(SPLIT_LEN, dtype=np.float32)

    print(f"bench start cfg={args.cfg} data={args.data_root} steps={args.steps} max={args.max_size} online={args.online} runs={runs}")
    t0 = time.perf_counter()
    for _ in range(runs):
        sdk.run_chunk(audio, CHUNKSIZE)

    deadline = time.perf_counter() + max(60, args.seconds * 4)
    while time.perf_counter() < deadline:
        with sink.lock:
            n = sink.n
            last = sink.last
        if n >= runs * CHUNKSIZE[1]:
            break
        if last and time.perf_counter() - last > 10:
            break
        time.sleep(0.2)

    dt = time.perf_counter() - t0
    with sink.lock:
        n = sink.n
    print(f"frames={n} expected={runs * CHUNKSIZE[1]} elapsed={dt:.2f}s fps={n / max(dt, 1e-6):.2f}")
    try:
        sdk.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
