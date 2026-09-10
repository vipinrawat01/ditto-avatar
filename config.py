###############################################################################
#  Config parsing — CLI args + YAML config
###############################################################################

import argparse
import json
import os

try:
    import yaml
    _has_yaml = True
except ImportError:
    _has_yaml = False


def str_or_int(value):
    """Try to convert to int, fall back to str on failure."""
    try:
        return int(value)
    except ValueError:
        return value


def _yaml_to_args(yaml_cfg):
    """Convert YAML dict keys into argparse-compatible `--key` dest names.

    argparse dest rules: `--model` -> `model`, `--push-url` -> `push_url`.
    This function accepts both key styles:
      - model / batch_size        -> passed through as-is
      - model-name / batch-size   -> converted to model_name / batch_size
    """
    result = {}
    for k, v in yaml_cfg.items():
        dest = k.replace('-', '_')
        result[dest] = v
    return result


def parse_args():
    """Parse CLI args, with an optional YAML config file overriding defaults.

    Priority: CLI args > YAML config file > add_argument(default=...)
    """
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")

    # ─── Config file ───────────────────────────────────────────────────
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='YAML config file path (set to empty string to skip)')

    # ─── Audio ─────────────────────────────────────────────────────────
    parser.add_argument('--fps', type=int, default=25, help="video fps, must be 25")
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    # ─── Display ───────────────────────────────────────────────────────
    # parser.add_argument('--W', type=int, default=450, help="GUI width")
    # parser.add_argument('--H', type=int, default=450, help="GUI height")

    # ─── Avatar model ──────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='wav2lip',
                        help="avatar model: musetalk/wav2lip/ultralight")
    parser.add_argument('--avatar_id', type=str, default='wav2lip256_avatar1',
                        help="avatar id in data/avatars")
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")
    parser.add_argument('--modelres', type=int, default=192)
    parser.add_argument('--modelfile', type=str, default='')

    # ─── Custom actions & multi-avatar ─────────────────────────────────
    parser.add_argument('--customvideo_config', type=str, default='',
                        help="custom action json")

    # ─── TTS ───────────────────────────────────────────────────────────
    parser.add_argument('--tts', type=str, default='edgetts',
                        help="tts plugin: edgetts/gpt-sovits/cosyvoice/fishtts/tencent/doubao/indextts2/azuretts/qwentts")
    parser.add_argument('--REF_FILE', type=str, default=None,
                        help="reference file name or voice model id (auto-picked by avatar_id when omitted)")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')

    # ─── Transport ─────────────────────────────────────────────────────
    parser.add_argument('--transport', type=str, default='webrtc',
                        help="output: rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=5)
    parser.add_argument('--listenport', type=int, default=8010,
                        help="web listen port")

    # ─── Load YAML config file ─────────────────────────────────────────
    if _has_yaml:
        # First do a partial parse to grab only the --config value
        tmp_opt, _ = parser.parse_known_args()
        config_path = tmp_opt.config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_cfg = yaml.safe_load(f)
            if yaml_cfg and isinstance(yaml_cfg, dict):
                yaml_defaults = _yaml_to_args(yaml_cfg)
                parser.set_defaults(**yaml_defaults)
    else:
        print("[config] PyYAML not installed, skipping YAML config file loading. "
              "Install: pip install pyyaml")

    # ─── Final CLI parse ───────────────────────────────────────────────
    opt = parser.parse_args()

    # ─── Post-processing ───────────────────────────────────────────────
    # When REF_FILE is not given, auto-pick an edgetts voice by avatar_id.
    # Note: 'woman' contains 'man', so check 'woman' first.
    if opt.REF_FILE is None:
        if opt.tts == 'elevenlabs':
            if 'woman' in opt.avatar_id:
                opt.REF_FILE = 'SEWXl8lPSO01tdGbWECX'
            elif 'man' in opt.avatar_id:
                opt.REF_FILE = 'aSXZu6bgEOS8MXVRzjPi'
            else:
                opt.REF_FILE = 'SEWXl8lPSO01tdGbWECX'
        elif 'woman' in opt.avatar_id:
            opt.REF_FILE = 'en-US-JennyNeural'
        elif 'man' in opt.avatar_id:
            opt.REF_FILE = 'en-US-GuyNeural'
        else:
            opt.REF_FILE = 'zh-CN-YunxiaNeural'  # ponytail: old default fallback

    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r') as f:
            opt.customopt = json.load(f)

    return opt
