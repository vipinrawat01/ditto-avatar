from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_avatar_frontend_contract():
    html = (ROOT / "web" / "index-en.html").read_text(encoding="utf-8")
    assert all(metric in html for metric in (
        "KB first:", "KB done:", "TTS first:", "Avatar first:"
    ))
    assert 'id="latencyFirst"' not in html
    assert "renderer.em = token => token.text" in html
    assert "marked.parse(markdown, { renderer })" in html
    assert "content.replaceChildren()" in html
    assert "className = 'chat-media'" in html
    assert "(?:mp4|mov|webm)" in html
    assert "https://www.youtube-nocookie.com/embed/" in html
    assert "function youtubeEmbedUrl(url)" in html
    assert "if (!youtube) content.appendChild(createLinkQr(url))" in html
    assert "function createLinkQr(url)" in html
    assert "className = 'link-qr'" in html
    assert "'/api/qr?data=' + encodeURIComponent(href)" in html
    assert "const CHAT_URL_RE = /`?((?:https?:\\/\\/|www\\.)" in html
    assert ".replace(/(?:https?:\\/\\/|www\\.)\\S+/g, ' ')" in html
    assert 'id="landingView"' in html
    assert 'id="landingVideo"' in html
    assert 'id="landingAvatarSelect"' not in html
    assert 'src="/api/avatar-idle/ditto_man?v=idle-1"' in html
    assert 'id="conversationView" class="main-layout" hidden' in html
    assert "function enterConversationAndTalk()" in html
    # A cold Ditto session must never block microphone capture: no awaited
    # ensureSession() anywhere on the path from a mic press to toggleMic().
    landing_talk = html.split("async function enterConversationAndTalk()", 1)[1].split("\n}", 1)[0]
    assert "await ensureSession()" not in landing_talk
    assert "await startVoiceMode()" in landing_talk
    voice_mode = html.split("async function startVoiceMode()", 1)[1].split("\n}", 1)[0]
    assert "await ensureSession()" not in voice_mode
    assert "await toggleMic()" in voice_mode
    assert "classList.remove('composer-open')" in voice_mode
    assert "function enterConversationWithPrompt(prompt)" in html
    assert "addBubble('user', prompt)" in html
    assert "await ensureSession()" in html
    assert "await sendText(prompt, false)" in html
    assert "async function sendText(textOverride = null, showUserBubble = true)" in html
    assert "const text = (textOverride == null ? el.value : textOverride).trim()" in html
    assert "function endConversation()" in html
    assert "function updateLandingSource()" in html
    assert "function setPreviewSource(video, sourceUrl)" in html
    assert "function avatarPreviewUrl(avatarId)" in html
    assert "window.location.protocol === 'file:'" in html
    assert "'../assets/idle.mp4'" in html
    assert "'../assets/woman.mp4'" in html
    assert "'/api/avatar-idle/' + encodeURIComponent(avatarId)" in html
    assert "animation: prompt-scroll 26s linear infinite" in html
    assert "animation-duration: 28s" in html
    assert "setPreviewSource(document.getElementById('video'), sourceUrl)" in html
    assert "Press to Talk" in html
    assert 'class="landing-prompts"' in html
    assert html.count('class="prompt-track"') == 2
    assert 'class="avatar-end-button"' not in html
    assert 'class="avatar-voice-control"' in html
    assert 'class="conversation-exit"' in html
    assert 'onclick="openEndSessionDialog()" title="Exit conversation"' in html
    assert 'id="endSessionDialog"' in html
    assert 'Are you sure you want to leave?' in html
    assert 'Continue Chatting' in html
    assert 'End Session' in html
    assert 'id="experienceRating"' in html
    assert 'Please rate your experience' in html
    assert 'Tap a star to rate' in html
    assert 'function scheduleExperienceRating()' in html
    assert '}, 8000);' in html
    assert "if (!speaking && wasSpeaking) scheduleExperienceRating();" in html
    assert "if (who === 'bot' && text.trim()) experienceRatingEligible = true;" in html
    assert "function resetExperienceRating()" in html
    assert "resetExperienceRating();" in html
    assert "function openEndSessionDialog()" in html
    assert "function closeEndSessionDialog()" in html
    assert "document.getElementById('chatResponse').replaceChildren()" in html
    assert "document.getElementById('landingView').hidden = false" in html
    end_conversation = html.split("function endConversation()", 1)[1].split("// --- Voice input", 1)[0]
    assert "interrupt()" in end_conversation
    assert "chatSessionId = createChatSessionId();" in end_conversation
    assert "pc.close()" not in end_conversation
    assert "setSessionId(null)" not in end_conversation
    assert "peer.close()" not in end_conversation
    assert "setSessionId(null)" not in end_conversation
    assert "video.srcObject = null" not in end_conversation
    assert "await interrupt()" not in end_conversation
    assert "if (sessionEnded || reconnectTimer) return" in html
    assert "className = 'bubble-feedback'" in html
    assert ".bubble-feedback { position: absolute; right: 9px; bottom: 5px;" in html
    assert "link.textContent = 'Open link'" not in html
    assert "feedback.hidden = true" in html
    assert "feedback.hidden = !text.trim()" in html
    assert 'data-feedback="up"' in html
    assert 'data-feedback="down"' in html
    assert "bi-hand-thumbs-up-fill" in html
    assert "bi-hand-thumbs-down-fill" in html
    assert "loadAvatarList().finally(() => start())" in html
    # Agent chat streams independently from the expensive WebRTC avatar build.
    assert "let chatSessionId = createChatSessionId();" in html
    assert "function createChatSessionId()" in html
    assert "function ensureSession()" in html
    assert "return new Promise(resolve => sessionWaiters.push(resolve));" in html
    prompt_flow = html.split("async function enterConversationWithPrompt(prompt)", 1)[1].split("\n}", 1)[0]
    assert "await ensureSession()" not in prompt_flow
    send_text = html.split("async function sendText(textOverride = null, showUserBubble = true)", 1)[1].split("// --- Interrupt", 1)[0]
    assert "ensureSession();" in send_text
    assert "await ensureSession()" not in send_text
    assert "session_id: chatSessionId" in send_text
    queue_speak = html.split("function queueSpeak(text, doInterrupt, pauseMs = 520, final = true)", 1)[1].split("// Where to cut", 1)[0]
    assert "await ensureSession();" in queue_speak
    assert "if (gen !== speakGen || !sessionid) return;" in queue_speak
    assert "row.hidden = true" in html
    assert ".latency-row { display: flex; flex-wrap: wrap;" in html
    assert "el.parentElement.hidden = false" in html
    assert "const interruptPromise = interrupt()" in html
    assert (
        "const CHAT_AGENT_ID = "
        "'DEVELOPMENT_HPB_17_215536fc419d4d45a6148239df3b1ba8';" in html
    )
    assert "const INITIAL_AVATAR_ID = 'ditto_man';" in html
    assert "avatars.includes(INITIAL_AVATAR_ID)" in html
    assert "id === preferred" in html
    assert "id.replace(/^ditto_/, 'avatar_')" in html
    assert "' (default)'" not in html
    assert "agent_id: CHAT_AGENT_ID" in html
    ditto_env = (ROOT / "docker" / "ditto-env.sh").read_text(encoding="utf-8")
    assert (
        "CHAT_AGENT_ID=${CHAT_AGENT_ID:-"
        "DEVELOPMENT_HPB_17_215536fc419d4d45a6148239df3b1ba8}" in ditto_env
    )
    assert "fetch('/avatar_timing'" in html
    assert '<audio id="remoteAudio" autoplay playsinline hidden></audio>' in html
    assert "stream.addTrack(evt.track)" in html
    assert "if (evt.track.kind === 'audio')" in html
    assert "remoteAudio.srcObject = new MediaStream([evt.track]);" in html
    assert "remoteAudio.play().catch(error => console.warn('remote audio playback blocked', error));" in html
    assert "html, body {" in html
    assert "overflow: hidden;" in html
    assert "height: 100dvh" in html
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)" in html
    assert "font-size: 16px; line-height: 1.4;" in html
    assert "color: var(--chat-fg); font-size: 16px;" in html
    # Right pane stays white; the kiosk reference contributes layout only.
    assert "--chat-bg: #fff;" in html
    assert "background: var(--chat-bg) !important" in html
    assert "--chat-bubble-bot: #f1f3f7;" in html
    # Landing action cluster: info / Ask a Question / mic, per the reference screens.
    landing = html.split('class="landing-actions"', 1)[1].split("</section>", 1)[0]
    assert landing.count('class="round-button"') == 2
    assert 'class="landing-ask"' in landing
    assert 'onclick="enterConversationAndType()"' in landing
    assert 'onclick="enterConversationAndTalk()"' in landing
    assert 'class="landing-talk"' not in html
    assert "function enterConversationAndType()" in html
    assert "function openInfoDialog()" in html
    assert 'id="infoDialog"' in html
    assert 'class="chat-brand"' in html
    assert "#conversationView[hidden] { display: none !important; }" in html
    # Text input is a full-width strip in the app's second grid row, opened on demand.
    assert "grid-template-rows: minmax(0, 1fr) auto" in html
    assert "#app.composer-open .chat-composer { display: flex; }" in html
    assert "function openComposer()" in html
    # Entering the conversation defaults to text input.
    assert "app.classList.add('conversation-active', 'composer-open')" in html
    assert "app.classList.remove('conversation-active', 'composer-open')" in html
    assert "#avatarControl" not in html
    # The stage control is a rounded rectangle; the legacy round blue #btnMic is gone.
    assert "#btnMic.voice-control { flex: 0 0 auto;" in html
    assert "border-radius: 18px" in html
    assert "background: #1597e5" not in html
    assert "#btnMic.listening" not in html
    assert "Press to Stop Recording" in html
    assert "voice-control-icon { display: inline-flex; width: 38px; height: 38px;" in html
    # Info panel copy is the HPB/Voncierge briefing, not a generic how-to.
    assert "Health Promotion Board (HPB)" in html
    assert "Receive answers grounded in HPB-approved oral health information" in html
    assert "https://voncierge.ai/" in html
    assert 'class="end-session-dialog info-dialog"' in html
    assert ".info-dialog { width: min(620px, 100%); aspect-ratio: auto;" in html
    assert html.index('class="chat-toolbar"') < html.index('id="chatResponse"')
    assert html.index('id="chatResponse"') < html.index('class="chat-composer"')
    assert html.index('class="chat-composer"') < html.index("<script>")
    assert "width: min(420px, 100%); padding: 24px 30px" in html
    assert "font-size: 1.28rem" in html
    assert "font-size: 0.9rem" in html
    assert ".end-session-actions { display: flex; flex-wrap: wrap; justify-content: center;" in html
    assert "#chatResponse { flex: 1;" in html
    assert "overflow-y: auto !important" in html
    assert "background: #fff !important" in html
    assert 'id="btnMic" class="voice-control" type="button" onclick="handleVoiceControl()"' in html
    assert html.count('id="btnMic"') == 1
    avatar_stage = html.split('class="mt-3 avatar-stage"', 1)[1].split('<div class="chat-pane"', 1)[0]
    composer = html.split('<div class="chat-composer">', 1)[1].split('</div>', 1)[0]
    assert 'id="btnMic"' in avatar_stage
    # Strip has one stateful action: mic while idle, Stop while text/voice output is active.
    assert 'id="btnMic"' not in composer
    assert 'id="txtMessage"' in composer
    assert "sendText()" in composer          # Enter key handler
    assert composer.count("<button") == 1
    assert 'class="composer-mic"' in composer
    assert 'id="composerAction"' in composer
    assert 'onclick="handleComposerAction()"' in composer
    assert "function renderComposerAction()" in html
    assert "async function stopTextResponse()" in html
    assert "activeChatAbortController.abort()" in html
    assert "const controller = new AbortController();" in send_text
    assert "signal: controller.signal," in send_text
    assert "if (e.name === 'AbortError') return;" in send_text
    assert "bi-trash" not in html
    assert "composer-send" not in html
    assert 'class="btn-outline-custom' not in html
    # Text mode and voice mode are exclusive.
    assert "#app.composer-open .avatar-voice-control { display: none; }" in html
    assert "async function startVoiceMode()" in html
    assert "bi-keyboard-fill" in avatar_stage
    assert 'onclick="openComposer()"' in avatar_stage
    assert "function renderMicButton()" in html
    assert "function drawVoiceBars()" in html
    assert "function stopVoiceVisualization()" in html
    assert "document.querySelectorAll('.voice-bar')" in html
    assert html.count('class="voice-bar"') == 12
    assert "const AUTO_SUBMIT_SILENCE_MS = 1200;" in html
    assert "const VOICE_ACTIVITY_RMS = 0.018;" in html
    assert "micAnalyser.getByteTimeDomainData(samples);" in html
    assert "async function handleVoiceControl()" in html
    assert "micProcessing = false" in html
    assert "micStarting = false" in html
    assert "Processing..." in html
    assert "Starting microphone..." in html
    assert "btn.disabled = mode === 'processing' || mode === 'starting'" in html
    assert "if (avatarSpeaking)" in html
    assert "setInterval(syncAvatarSpeaking, 250)" in html
    assert "micSource.connect(micAnalyser)" in html
    toggle_mic = html.split("async function toggleMic()", 1)[1].split(
        "function stopMic", 1
    )[0]
    assert "Please connect WebRTC first" not in toggle_mic
    assert "window.SpeechRecognition || window.webkitSpeechRecognition" in toggle_mic
    assert "if (micStarting) return;" in toggle_mic
    assert "micStarting = true;" in toggle_mic
    assert "recognition.continuous = true" in toggle_mic
    assert "recognition.interimResults = true" in toggle_mic
    assert "recognition.lang = 'en-US'" in toggle_mic
    assert "recognition.onresult = event =>" in toggle_mic
    assert "echoCancellation: true" in toggle_mic
    assert "noiseSuppression: true" in toggle_mic
    assert "autoGainControl: true" in toggle_mic
    assert "'/api/asr'" not in html
    assert '<select id="txtType" class="visually-hidden"' in html
    assert ".chat-bubble { position: relative; width: fit-content;" in html
    assert "if (connecting || (pc && ['new', 'connecting', 'connected'].includes(pc.connectionState))) return;" in html
    assert "if (pc !== peer) return;" in html
    assert "scheduleReconnect(5000)" in html
    assert "function semanticBoundary(s, force = false)" in html
    semantic = html.split("function semanticBoundary", 1)[1].split("function addBubble", 1)[0]
    assert "HARD_MAX" not in semantic
    assert "lastSpace" not in semantic
    assert "s.startsWith(' - ', i)" in semantic
    assert "pauseMs: 280" in semantic
    assert "paragraph ? 900 : 520" in html
    assert "pauseMs: 800" in html
    assert "'excl'" in html
    assert "pause_ms: pauseMs" in html
    assert "final, sessionid" in html
    assert "queueSpeak('', false, 20, true)" in html


def test_avatar_source_endpoint_is_limited_to_source_video():
    routes = (ROOT / "server" / "routes.py").read_text(encoding="utf-8")
    assert "async def avatar_source_media(request):" in routes
    assert 're.fullmatch(r"[A-Za-z0-9_-]+", avatar_id)' in routes
    assert 'for filename in ("source.mp4", "source.webm", "source.mov")' in routes
    assert 'APP_ROOT = Path(__file__).resolve().parent.parent' in routes
    assert '"ditto_woman": APP_ROOT / "assets" / "woman.mp4"' in routes
    assert '"ditto_man": APP_ROOT / "assets" / "source.mp4"' in routes
    assert 'app.router.add_get("/api/avatar-source/{avatar_id}", avatar_source_media)' in routes


def test_avatar_idle_endpoint_is_limited_to_idle_video():
    routes = (ROOT / "server" / "routes.py").read_text(encoding="utf-8")
    assert "async def avatar_idle_media(request):" in routes
    assert 'idle = avatar_dir / "idle.mp4"' in routes
    assert 'app.router.add_get("/api/avatar-idle/{avatar_id}", avatar_idle_media)' in routes


def test_qr_codes_are_generated_locally():
    routes = (ROOT / "server" / "routes.py").read_text(encoding="utf-8")
    requirements = (ROOT / "docker" / "requirements.txt").read_text(encoding="utf-8")
    assert "async def qr_code(request):" in routes
    assert 'app.router.add_get("/api/qr", qr_code)' in routes
    assert 'content_type="image/png"' in routes
    assert "qrcode[pil]==8.2" in requirements


def test_ditto_defaults_to_fast_high_resolution_rendering():
    script = (ROOT / "docker" / "ditto-env.sh").read_text(encoding="utf-8")
    # Sourced, not duplicated: a shell in JupyterLab must be able to reproduce
    # the server's exact rendering parameters before regenerating idle.mp4.
    start = (ROOT / "docker" / "start.sh").read_text(encoding="utf-8")
    assert 'source "$APP_ROOT/docker/ditto-env.sh"' in start
    assert "DITTO_STEPS" not in start, "defaults must live in ditto-env.sh only"
    assert "DITTO_STEPS=${DITTO_STEPS:-5}" in script
    assert "DITTO_EXP=${DITTO_EXP:-0.63}" in script
    assert "DITTO_MAX_SIZE=${DITTO_MAX_SIZE:-1280}" in script
    assert "DITTO_LIP_RESPONSE=${DITTO_LIP_RESPONSE:-1.4}" in script
    assert "DITTO_PAUSE_CLOSE_MS=${DITTO_PAUSE_CLOSE_MS:-120}" in script
    assert "DITTO_FEED_CAP=${DITTO_FEED_CAP:-20}" in script
    assert "DITTO_START_BUFFER=${DITTO_START_BUFFER:-6}" in script
    assert "DITTO_HOLD=${DITTO_HOLD:-0.10}" in script
    assert "DITTO_TAIL_MS=${DITTO_TAIL_MS:-500}" in script
    assert "DITTO_IDLE_FADE_MS" not in script
    assert "DITTO_AV_OFFSET_MS=${DITTO_AV_OFFSET_MS:-220}" in script
    assert "DITTO_FINAL_HOLD_MS=${DITTO_FINAL_HOLD_MS:-240}" in script
    assert "DITTO_GENERATE_IDLE=${DITTO_GENERATE_IDLE:-0}" in script
    assert "DITTO_USE_GENERATED_IDLE=${DITTO_USE_GENERATED_IDLE:-0}" in script
    assert "ASR_LANGUAGE=${ASR_LANGUAGE:-English}" in script
    assert "ASR_BACKEND=${ASR_BACKEND:-browser}" in script


def test_qwen_asr_is_restricted_to_english_by_default():
    server = (ROOT / "server" / "asr_server.py").read_text(encoding="utf-8")
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert '_ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "English").strip() or None' in server
    assert "language=_ASR_LANGUAGE" in server
    assert "ASR_LANGUAGE=English" in env


def test_browser_stt_disables_local_asr_by_default():
    routes = (ROOT / "server" / "routes.py").read_text(encoding="utf-8")
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert 'os.environ.get("ASR_BACKEND", "browser").lower() == "local"' in routes
    assert "[ASR] Browser SpeechRecognition enabled (en-US); local model disabled" in routes
    assert "ASR_BACKEND=browser" in env


def test_generated_idle_is_optional_and_preserves_original():
    start = (ROOT / "docker" / "start.sh").read_text(encoding="utf-8")
    avatar = (ROOT / "avatars" / "ditto_avatar.py").read_text(encoding="utf-8")
    generator = (ROOT / "scripts" / "ditto_make_idle.py").read_text(encoding="utf-8")

    assert 'if [[ "$DITTO_GENERATE_IDLE" == "1" ]]' in start
    assert "WARNING: generated idle failed; using original idle.mp4" in start
    assert 'os.environ.get("DITTO_USE_GENERATED_IDLE", "0") == "1"' in avatar
    assert '("idle.generated.mp4", "idle.mp4") if use_generated else ("idle.mp4",)' in avatar
    assert 'not os.path.exists(idle_path + ".json")' in avatar
    assert "neutralize_sdk_source_lips(sdk, original_idle)" in generator
    assert 'sdk.setup(original_idle, "/tmp/ditto_make_idle_dummy.mp4"' in generator
    assert "GENERATOR_VERSION = 3" in generator
    assert 'original_idle = os.path.join(avatar_dir, "idle.mp4")' in generator
    assert "os.replace(temporary_out, out)" in generator
    assert "os.replace(temporary_out, original_idle)" not in generator


def test_ditto_exposes_timing_events():
    avatar = (ROOT / "avatars" / "ditto_avatar.py").read_text(encoding="utf-8")
    routes = (ROOT / "server" / "routes.py").read_text(encoding="utf-8")
    assert "self._tts_start_seq += 1" in avatar
    assert "self._avatar_start_seq += 1" in avatar
    assert 'app.router.add_post("/avatar_timing", avatar_timing)' in routes


def test_elevenlabs_forwards_pcm_while_streaming():
    tts = (ROOT / "tts" / "elevenlabs_tts.py").read_text(encoding="utf-8")
    assert 'raw = b"".join(chunks)' not in tts
    assert "for pcm_chunk in chunks:" in tts
    # Frames leave while the stream is still arriving. A fixed, short rolling
    # tail is retained so the terminal phoneme can carry a lip-closure hint.
    assert "queue_frame(frame)" in tts
    assert "len(phoneme_tail) > _PHONEME_TAIL_FRAMES" in tts
    assert "self._emit([phoneme_tail.pop(0)], text, textevent, gain)" in tts
    assert "output = _level_frame(frame, gain)" in tts
    assert "elevenlabs output level:" in tts
    assert "if len(held) >= _ESTIMATE_FRAMES:" in tts
    assert "re.split" not in tts
    assert "for _ in range(3)" not in tts
    assert "elevenlabs first audio:" in tts
    assert "previous_text=self._previous_text or None" in tts
    assert "for index in range((pause_ms + 19) // 20):" in tts
    assert '"segment_end"' in tts
