import os
import json
import requests
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

API_URL = os.getenv("CHAT_API_URL", "https://voncierge-lang-agent.hipster-virtual.com/chat/sse")
AGENT_ID = os.getenv(
    "CHAT_AGENT_ID", "DEVELOPMENT_HPB_17_215536fc419d4d45a6148239df3b1ba8"
)

# Strip markdown symbols that TTS would read aloud
_STRIP = str.maketrans("", "", "*#`")


def llm_response(message, avatar_session: 'BaseAvatar', datainfo: dict = {}):
    """Send user input to the voncierge SSE API; push complete sentences to TTS as they stream in."""
    try:
        session_id = "1212"
        resp = requests.post(
            API_URL,
            json={"message": message, "session_id": session_id, "agent_id": AGENT_ID},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()

        buffer = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            try:
                obj = json.loads(line[5:].strip())
            except Exception:
                continue
            buffer += obj.get("data", "")

            # Flush a sentence on each sentence-ending punctuation to reduce latency
            while True:
                idx = next((i for i, c in enumerate(buffer) if c in ".!?。！？\n"), -1)
                if idx < 0:
                    break
                seg, buffer = buffer[:idx + 1], buffer[idx + 1:]
                seg = seg.translate(_STRIP).strip()
                if seg:
                    logger.info(f"api seg: {seg}")
                    avatar_session.put_msg_txt(seg, datainfo)

            if obj.get("final_token"):
                break

        tail = buffer.translate(_STRIP).strip()
        if tail:
            avatar_session.put_msg_txt(tail, datainfo)

    except Exception:
        logger.exception('chat api exception:')
        return
2
