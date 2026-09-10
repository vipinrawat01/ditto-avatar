###############################################################################
#  Server routes — API routes with unified exception handling
###############################################################################

import json
import asyncio
from io import BytesIO
import os
import re
from pathlib import Path
from aiohttp import web
import qrcode

from utils.logger import logger


APP_ROOT = Path(__file__).resolve().parent.parent


# ─── Route helper functions ─────────────────────────────────────────────────

def json_ok(data=None):
    """Return a success JSON response"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """Return an error JSON response"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager
from server.avatar_routes import setup_avatar_routes

def get_session(request, sessionid: str):
    """Get the session instance from the app"""
    return session_manager.get_session(sessionid)


# ─── Route handler functions ────────────────────────────────────────────────

async def human(request):
    """Text input (echo/chat mode), supports voice/emotion parameters"""
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):  # pass through tts parameters (voice, emotion, etc.)
            datainfo['tts'] = params.get('tts')
        try:
            pause_ms = int(params.get('pause_ms', 0))
        except (TypeError, ValueError):
            pause_ms = 0
        if pause_ms:
            # The browser selects semantic pauses at sentence, paragraph, and
            # list boundaries. Bound them before they reach the realtime path.
            datainfo['pause_ms'] = max(20, min(pause_ms, 1000))
        datainfo['final'] = bool(params.get('final', True))

        if params['type'] == 'echo':
            if params['text'].strip() or datainfo['final']:
                avatar_session.put_msg_txt(params['text'], datainfo)
        elif params['type'] == 'chat':
            llm_response = request.app.get("llm_response")
            if llm_response:
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params['text'], avatar_session, datainfo
                )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """Interrupt the current speech"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as e:
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """Upload an audio file"""
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.put_audio_file(filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def set_audiotype(request):
    """Set a custom state (action orchestration)"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """Recording control"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params['type'] == 'start_record':
            avatar_session.start_recording()
        elif params['type'] == 'end_record':
            avatar_session.stop_recording()
        return json_ok()
    except Exception as e:
        logger.exception('record exception:')
        return json_error(str(e))


async def is_speaking(request):
    """Query whether currently speaking"""
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())


async def avatar_timing(request):
    params = await request.json()
    avatar_session = get_session(request, params.get('sessionid', ''))
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data={
        "tts_seq": getattr(avatar_session, "_tts_start_seq", 0),
        "avatar_seq": getattr(avatar_session, "_avatar_start_seq", 0),
    })


async def list_avatars(request):
    """List avatar IDs compatible with the RUNNING model.

    data/avatars holds folders for several backends (ditto needs source.*;
    musetalk has avator_info.json + full_imgs/). Listing them all lets the UI
    offer avatars the current model can't load, so filter to the running model."""
    import os
    base = os.path.join('data', 'avatars')
    opt = request.app.get("opt")
    model = getattr(opt, "model", "") if opt else ""
    items = []
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            p = os.path.join(base, d)
            if not os.path.isdir(p):
                continue
            if model == "ditto":
                # ditto needs a source image/video; skip musetalk/others
                if not any(os.path.exists(os.path.join(p, f"source.{e}"))
                           for e in ("mp4", "png", "jpg", "jpeg")):
                    continue
            items.append(d)
    return json_ok(data={"avatars": items})


async def avatar_source_media(request):
    """Serve only a selected avatar's source video for the kiosk landing view."""
    avatar_id = request.match_info.get("avatar_id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", avatar_id):
        raise web.HTTPNotFound()

    bundled_previews = {
        "ditto_woman": APP_ROOT / "assets" / "woman.mp4",
        "ditto_man": APP_ROOT / "assets" / "source.mp4",
    }
    bundled_source = bundled_previews.get(avatar_id)
    if bundled_source and bundled_source.is_file():
        return web.FileResponse(bundled_source)

    avatars_root = APP_ROOT / "data" / "avatars"
    avatar_dir = (avatars_root / avatar_id).resolve()
    try:
        avatar_dir.relative_to(avatars_root.resolve())
    except ValueError:
        raise web.HTTPNotFound()

    for filename in ("source.mp4", "source.webm", "source.mov"):
        source = avatar_dir / filename
        if source.is_file():
            return web.FileResponse(source)
    raise web.HTTPNotFound()


async def avatar_idle_media(request):
    """Serve only a selected avatar's idle video for local preview states."""
    avatar_id = request.match_info.get("avatar_id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", avatar_id):
        raise web.HTTPNotFound()

    avatars_root = APP_ROOT / "data" / "avatars"
    avatar_dir = (avatars_root / avatar_id).resolve()
    try:
        avatar_dir.relative_to(avatars_root.resolve())
    except ValueError:
        raise web.HTTPNotFound()

    idle = avatar_dir / "idle.mp4"
    if idle.is_file():
        return web.FileResponse(idle)

    bundled_idle = {
        "ditto_man": APP_ROOT / "assets" / "idle.mp4",
        "ditto_woman": APP_ROOT / "assets" / "woman.mp4",
    }.get(avatar_id)
    if bundled_idle and bundled_idle.is_file():
        return web.FileResponse(bundled_idle)
    raise web.HTTPNotFound()


async def qr_code(request):
    """Generate a QR image locally so kiosk browsers need no third-party QR host."""
    data = request.query.get("data", "").strip()
    if not data or len(data) > 2048:
        raise web.HTTPBadRequest(text="A QR payload up to 2048 characters is required")

    try:
        code = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=2,
        )
        code.add_data(data)
        code.make(fit=True)
        image = code.make_image(fill_color="#101828", back_color="white")
        output = BytesIO()
        image.save(output, format="PNG")
    except Exception as exc:
        logger.exception("QR generation failed")
        raise web.HTTPBadRequest(text=f"Unable to generate QR code: {exc}")

    return web.Response(
        body=output.getvalue(),
        content_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


async def admin_config(request):
    """Admin: get global configuration parameters"""
    try:
        opt = request.app.get("opt")
        if opt:
            return json_ok(data={"config": vars(opt)})
        return json_error("Config not found")
    except Exception as e:
        logger.exception('admin_config exception:')
        return json_error(str(e))


async def admin_sessions(request):
    """Admin: get active sessions and their configuration"""
    try:
        sessions_info = []
        for sid, avatar_session in session_manager.sessions.items():
            if avatar_session:
                s_opt = getattr(avatar_session, 'opt', None)
                s_data = {
                    "sessionid": sid,
                    "speaking": avatar_session.is_speaking() if hasattr(avatar_session, 'is_speaking') else False,
                    "recording": getattr(avatar_session, 'recording', False),
                }
                if s_opt:
                    s_data.update({
                        "model": getattr(s_opt, "model", ""),
                        "avatar_id": getattr(s_opt, "avatar_id", ""),
                        "REF_FILE": getattr(s_opt, "REF_FILE", ""),
                        "transport": getattr(s_opt, "transport", ""),
                        "batch_size": getattr(s_opt, "batch_size", 0),
                        "customopt": getattr(s_opt, "customopt", []),
                    })
                sessions_info.append(s_data)
        return json_ok(data={"sessions": sessions_info})
    except Exception as e:
        logger.exception('admin_sessions exception:')
        return json_error(str(e))


async def ice_servers_route(request):
    """Return the ICE servers dynamically generated by Cloudflare for the frontend to use"""
    from server.turn import get_ice_servers
    return web.Response(
        content_type="application/json",
        text=json.dumps({"iceServers": get_ice_servers()}),
    )


# ─── Route registration ──────────────────────────────────────────────────────

def setup_routes(app):
    """Register all routes on the aiohttp app"""
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_post("/avatar_timing", avatar_timing)
    app.router.add_get("/api/avatars", list_avatars)
    app.router.add_get("/api/avatar-source/{avatar_id}", avatar_source_media)
    app.router.add_get("/api/avatar-idle/{avatar_id}", avatar_idle_media)
    app.router.add_get("/api/qr", qr_code)
    app.router.add_get("/api/admin/config", admin_config)
    app.router.add_get("/api/admin/sessions", admin_sessions)
    app.router.add_get("/api/iceservers", ice_servers_route)

    # Browser SpeechRecognition is the default STT. Keep the local endpoint as
    # an opt-in rollback path without loading Qwen/FunASR during normal startup.
    if os.environ.get("ASR_BACKEND", "browser").lower() == "local":
        try:
            from server.asr_server import asr_websocket_handler, is_funasr_available, warmup_async
            if is_funasr_available():
                app.router.add_get("/api/asr", asr_websocket_handler)
                warmup_async()
                logger.info("[ASR] Local ASR endpoint enabled at /api/asr")
            else:
                logger.info("[ASR] local ASR backend unavailable — /api/asr disabled")
        except Exception as e:
            logger.warning(f"[ASR] Failed to register ASR endpoint: {e}")
    else:
        logger.info("[ASR] Browser SpeechRecognition enabled (en-US); local model disabled")

    # Register avatar generation related routes
    setup_avatar_routes(app)

    app.router.add_static('/', path='web')
