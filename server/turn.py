###############################################################################
#  Cloudflare TURN — auto-generate short-lived ICE credentials from a
#  permanent API token, so they never need manual refresh.
#  Set CF_TURN_KEY_ID and CF_TURN_API_TOKEN in .env (both are permanent).
###############################################################################

import os
import json
import urllib.request

from utils.logger import logger

_FALLBACK = [{"urls": "stun:stun.cloudflare.com:3478"}]


def get_ice_servers(ttl: int = 86400):
    """Return a list of browser-compatible iceServer dicts.

    Fetches fresh TURN credentials from Cloudflare using the permanent
    key id + API token. Falls back to STUN-only if not configured or on error.
    """
    key_id = os.environ.get("CF_TURN_KEY_ID")
    token = os.environ.get("CF_TURN_API_TOKEN")
    if not key_id or not token:
        return _FALLBACK

    url = f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers"
    req = urllib.request.Request(
        url,
        data=json.dumps({"ttl": ttl}).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",  # Cloudflare WAF blocks default urllib UA (error 1010)
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        servers = data.get("iceServers")
        # Cloudflare returns a single object; normalize to a list
        if isinstance(servers, dict):
            servers = [servers]
        return servers or _FALLBACK
    except Exception as e:  # noqa: BLE001
        logger.warning("Cloudflare TURN fetch failed, falling back to STUN: %s", e)
        return _FALLBACK
