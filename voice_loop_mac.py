#!/usr/bin/env python3
"""Voice Loop — a minimal on-device voice agent. Mac M4 / Apple Silicon.

Moonshine (CPU) transcribes speech. Gemma 4 E4B (Metal) responds.
Kokoro TTS speaks the response. WebRTC AEC3 enables voice interrupt.

Usage:
    uv run voice_loop_mac.py                        # defaults (TTS + smart turn + AEC)
    uv run voice_loop_mac.py --no-tts               # text out only
    uv run voice_loop_mac.py --no-aec               # keypress interrupt only
    uv run voice_loop_mac.py --chime-loop           # chime + ticks while generating
"""

import argparse
import asyncio
import os
import queue
import re
import select
import sys
import tempfile
import termios
import threading
import time as _time
import tty
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sounddevice as sd
# Larger audio buffer via 'high' latency → more robust to MLX CPU saturation.
# NB: don't set sd.default.blocksize globally — a large blocksize on the TTS
# output stream introduces a mic-to-reference delay that misaligns AEC.
sd.default.latency = 'high'
import torch

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz (required by Silero VAD)
MAX_HISTORY = 10
CHIME_SR = 24000
_DIR = Path(__file__).parent

# Matches sentence-ending punctuation followed by whitespace.
# Used to split LLM output into sentences for early TTS dispatch.
_SENT_END = re.compile(r'(?<=[.!?])\s+')
# Accumulate short fragments (e.g. "Mr.") into the next sentence before dispatch.
_SENT_MIN_CHARS = 20
# Blanking window after a sentence ends: skip AEC during the room reverb tail to
# avoid treating residual echo as speech when the zero reference is fed to AEC3.
_GAP_BLANK_SAMPLES = int(0.15 * 16000)  # 150ms @ 16kHz


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, merging short fragments to avoid TTS artefacts."""
    parts, carry = [], ""
    for p in _SENT_END.split(text.strip()):
        p = p.strip()
        if not p:
            continue
        carry = (carry + " " + p).strip() if carry else p
        if len(carry) >= _SENT_MIN_CHARS:
            parts.append(carry)
            carry = ""
    if carry:
        parts.append(carry)
    return parts


def load_system_prompt(include_memory: bool = False) -> str:
    names = ("SOUL.md", "MEMORY.md") if include_memory else ("SOUL.md",)
    parts = [(_DIR / n).read_text().strip() for n in names if (_DIR / n).exists()]
    return "\n\n".join(p for p in parts if p)


def _fade_tone(freq, dur, amp=0.6):
    """Tone with raised-cosine (Hann) envelope — smooth fade in/out, no clicks."""
    n = int(dur * CHIME_SR)
    t = np.linspace(0, dur, n, dtype=np.float32)
    env = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    return amp * np.sin(2 * np.pi * freq * t) * env

def _silence(dur):
    return np.zeros(int(dur * CHIME_SR), dtype=np.float32)

def make_chime(duration=30.0, tick_every=1.5):
    """Two-tone chime + periodic short ticks. Single buffer → one sd.play()."""
    head = np.concatenate([_fade_tone(880, 0.09), _silence(0.03), _fade_tone(1320, 0.10)])
    # Short soft click-style tick (shorter and quieter than a beep)
    tick = _fade_tone(550, 0.04, amp=0.18)
    total = int(duration * CHIME_SR)
    buf = np.zeros(total, dtype=np.float32)
    buf[:len(head)] = head
    step = int(tick_every * CHIME_SR)
    for pos in range(len(head), total, step):
        end = min(pos + len(tick), total)
        buf[pos:end] = tick[:end - pos]
    return buf

def _lang_from_voice(v: str) -> str:
    """Infer Kokoro lang code from voice prefix.
    a* = US English, b* = UK English, e* = Spanish, f* = French,
    h* = Hindi, i* = Italian, j* = Japanese, p* = Portuguese, z* = Chinese."""
    prefix = v[:1] if len(v) > 1 and v[1] == '_' else ''
    return {
        'a': 'en-us', 'b': 'en-gb',
        'e': 'es', 'f': 'fr-fr', 'h': 'hi',
        'i': 'it', 'j': 'ja', 'p': 'pt-br', 'z': 'cmn',
    }.get(prefix, 'en-us')


def save_wav(audio, sr=SAMPLE_RATE):
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
    return path


def load_smart_turn():
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    model_path = os.path.join(tempfile.gettempdir(), "smart_turn_v3", "smart_turn_v3.2_cpu.onnx")
    if not os.path.exists(model_path):
        print("Downloading Smart Turn v3.2 model...", flush=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(
            "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx", model_path)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    def predict(audio_float32: np.ndarray) -> float:
        max_samples = 8 * SAMPLE_RATE
        audio_float32 = audio_float32[-max_samples:]
        features = extractor(
            audio_float32, sampling_rate=SAMPLE_RATE, max_length=max_samples,
            padding="max_length", return_attention_mask=False, return_tensors="np",
        )
        return float(session.run(None, {"input_features": features.input_features.astype(np.float32)})[0].flatten()[0])
    return predict

def _vad_prob(vad, chunk):
    p = vad(torch.from_numpy(chunk), SAMPLE_RATE)
    return p.item() if hasattr(p, "item") else p

def _get_ref_segment(tts_concat, pos, length):
    if pos >= len(tts_concat):
        return np.zeros(length, dtype=np.float32)
    seg = tts_concat[pos:pos + length]
    return np.concatenate([seg, np.zeros(length - len(seg), dtype=np.float32)]) if len(seg) < length else seg


def main():
    ap = argparse.ArgumentParser(description="Voice Loop — a minimal on-device voice agent (Mac)")
    B = argparse.BooleanOptionalAction
    ap.add_argument("--tts", action=B, default=True, help="Kokoro TTS output")
    ap.add_argument("--smart-turn", action=B, default=True, help="Smart Turn v3 endpoint detection")
    ap.add_argument("--aec", action=B, default=True, help="WebRTC AEC3 voice interrupt")
    ap.add_argument("--chime", action=B, default=True,
                    help="Chime on utterance + soft ticks while generating (default: on)")
    ap.add_argument("--memory", action="store_true",
                    help="Read/write MEMORY.md (auto-update durable facts, consolidate every 5 turns)")
    ap.add_argument("--audio-mode", action="store_true", help="Send audio directly to Gemma (experimental)")
    ap.add_argument("--model", default="mlx-community/gemma-4-E4B-it-4bit")
    ap.add_argument("--silence-ms", type=int, default=700)
    ap.add_argument("--record", nargs="?", const="", metavar="FILE",
                    help="Record mic to WAV for debugging (default: tmp/recording-TIMESTAMP.wav)")
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice")
    args = ap.parse_args()
    if args.record == "":
        tmp_dir = _DIR / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        args.record = str(tmp_dir / f"recording-{_time.strftime('%Y%m%d-%H%M%S')}.wav")
    silence_limit = max(1, int(args.silence_ms / (CHUNK_SAMPLES / SAMPLE_RATE * 1000)))

    print("Loading Silero VAD...", flush=True)
    from silero_vad import load_silero_vad
    vad = load_silero_vad(onnx=True)
    print("Loading Moonshine (transcription)...", flush=True)
    from moonshine_voice import Transcriber, get_model_for_language
    ms_path, ms_arch = get_model_for_language("en")
    moonshine = Transcriber(model_path=str(ms_path), model_arch=ms_arch)
    print(f"Loading {args.model} (first run downloads ~3GB)...", flush=True)
    from mlx_vlm import load, generate
    try:
        from mlx_vlm import stream_generate as _mlx_stream_generate
    except ImportError:
        _mlx_stream_generate = None
    model, processor = load(args.model)
    smart_turn = load_smart_turn() if args.smart_turn else None
    kokoro = None
    if args.tts:
        print("Loading Kokoro TTS...", flush=True)
        import subprocess
        try:
            prefix = subprocess.check_output(["brew", "--prefix", "espeak-ng"], text=True).strip()
            os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", f"{prefix}/lib/libespeak-ng.dylib")
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        from kokoro_onnx import Kokoro
        cache_dir = os.path.join(tempfile.gettempdir(), "kokoro_tts")
        model_file = os.path.join(cache_dir, "kokoro-v1.0.onnx")
        voices_file = os.path.join(cache_dir, "voices-v1.0.bin")
        if not os.path.exists(model_file):
            os.makedirs(cache_dir, exist_ok=True)
            import urllib.request
            base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
            print("  Downloading kokoro model (~300MB)...", flush=True)
            urllib.request.urlretrieve(f"{base}/kokoro-v1.0.onnx", model_file)
            urllib.request.urlretrieve(f"{base}/voices-v1.0.bin", voices_file)
        kokoro = Kokoro(model_file, voices_file)

    make_aec_processor = None
    if args.aec:
        from livekit.rtc import AudioFrame
        from livekit.rtc.apm import AudioProcessingModule
        WF = 160  # 10ms @ 16kHz
        def _to_i16(x):
            s = (x * 32767).clip(-32768, 32767).astype(np.int16)
            return np.pad(s, (0, max(0, WF - len(s)))) if len(s) < WF else s
        def _frame(b):
            return AudioFrame(b.tobytes(), sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=WF)
        def make_aec_processor():
            apm = AudioProcessingModule(echo_cancellation=True, noise_suppression=True)
            def process(mic, ref):
                cleaned = np.zeros_like(mic)
                for i in range(0, len(mic), WF):
                    mic_f = _frame(_to_i16(mic[i:i+WF]))
                    apm.process_reverse_stream(_frame(_to_i16(ref[i:i+WF])))
                    apm.process_stream(mic_f)
                    cleaned[i:i+WF] = (np.frombuffer(bytes(mic_f.data), dtype=np.int16).astype(np.float32) / 32767)[:len(mic[i:i+WF])]
                return cleaned
            return process
        print("  AEC: WebRTC AEC3 (LiveKit APM)")
    executor = ThreadPoolExecutor(max_workers=1)
    # --chime-loop: single buffer (chime + ticks), one sd.play call
    # --chime only: just the chime
    chime_sound = make_chime() if args.chime else None
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    record_buf: list[np.ndarray] | None = [] if args.record else None

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        chunk = indata[:, 0].copy()
        if record_buf is not None:
            record_buf.append(chunk)
        audio_q.put(chunk)

    def drain_audio_q():
        while not audio_q.empty():
            audio_q.get_nowait()

    def transcribe(audio_data):
        return " ".join(l.text for l in moonshine.transcribe_without_streaming(
            audio_data.tolist(), SAMPLE_RATE).lines if l.text).strip()

    def llm_generate(messages, max_tokens=200, temperature=0.7, **kwargs):
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        r = generate(model, processor, prompt, max_tokens=max_tokens,
                     temperature=temperature, repetition_penalty=1.2, verbose=False, **kwargs)
        return r.text if hasattr(r, "text") else str(r)

    def stream_sentences(messages, max_tokens=200, temperature=0.7):
        """Yield sentences as LLM generates them. LLM runs in a background thread.

        If mlx_vlm.stream_generate is available, sentences are dispatched as each
        one completes during token generation. Otherwise falls back to full
        generation + sentence split, which still allows TTS to start sooner by
        running LLM off the main thread.

        Short fragments (< _SENT_MIN_CHARS) are merged with the next sentence so
        abbreviations like "Mr." don't become standalone TTS invocations.
        """
        q: queue.Queue[str | None] = queue.Queue()
        cancel = threading.Event()

        def _worker():
            try:
                if _mlx_stream_generate is not None:
                    token_buf, carry = "", ""
                    prompt = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    for result in _mlx_stream_generate(
                        model, processor, prompt,
                        max_tokens=max_tokens, temperature=temperature,
                        repetition_penalty=1.2, verbose=False,
                    ):
                        if cancel.is_set():
                            return
                        token = result.text if hasattr(result, "text") else str(result)
                        token_buf += token
                        while True:
                            m = _SENT_END.search(token_buf)
                            if not m:
                                break
                            candidate = token_buf[: m.start() + 1].strip()
                            token_buf = token_buf[m.end() :]
                            carry = (carry + " " + candidate).strip() if carry else candidate
                            if len(carry) >= _SENT_MIN_CHARS:
                                q.put(carry)
                                carry = ""
                    # Flush remainder (merge with any carry)
                    remainder = (carry + " " + token_buf).strip() if token_buf.strip() else carry
                    if remainder:
                        q.put(remainder)
                else:
                    text = llm_generate(messages, max_tokens=max_tokens, temperature=temperature)
                    for s in _split_sentences(text) or [text]:
                        if cancel.is_set():
                            return
                        q.put(s)
            except Exception as e:
                print(f"  [LLM error: {e}]", file=sys.stderr)
            finally:
                q.put(None)

        threading.Thread(target=_worker, daemon=True).start()
        try:
            while True:
                s = q.get()
                if s is None:
                    return
                yield s
        finally:
            # Signal the worker to stop if the generator is abandoned mid-response
            # (e.g. barge-in interruption) so MLX doesn't keep generating.
            cancel.set()

    def speak_tts(text):
        samples, sr = kokoro.create(text, voice=args.voice, speed=1.0, lang=_lang_from_voice(args.voice))
        sd.play(samples, sr); sd.wait()

    _mem_path = _DIR / "MEMORY.md"

    def _read_memory():
        return _mem_path.read_text() if _mem_path.exists() else "# Memory\n"

    def _run_memory(prompt, max_tokens, temperature, label):
        try:
            return llm_generate(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature,
            ).strip()
        except Exception as e:
            print(f"  [{label} failed: {e}]", file=sys.stderr)
            return None

    def update_memory(heard, response):
        result = _run_memory(
            f"Current memory:\n{_read_memory()}\n\n"
            f"User said: {heard}\n\n"
            "Did the user state a new durable fact about themselves? "
            "If yes, output one short fact per line starting with '- '. "
            "If no, output ONLY: NONE. Do not invent facts.",
            max_tokens=60, temperature=0.2, label="memory update",
        )
        if result and "NONE" not in result.upper():
            lines = [l for l in result.splitlines() if l.strip().startswith("-")]
            if lines:
                with open(_mem_path, "a") as f:
                    f.write("\n" + "\n".join(lines) + "\n")
                print(f"  [memory +{len(lines)}]", flush=True)

    def consolidate_memory():
        if not _mem_path.exists():
            return
        result = _run_memory(
            f"Here is a memory file about a user:\n\n{_read_memory()}\n\n"
            "Rewrite it: merge duplicates, remove transient/session-specific "
            "items (questions asked, topics discussed, tests), keep only "
            "durable facts (identity, preferences, relationships, location, "
            "ongoing projects). Output the cleaned file, starting with '# Memory' "
            "followed by bullets starting with '- '. No explanation.",
            max_tokens=300, temperature=0.2, label="memory consolidation",
        )
        if result and result.startswith("# Memory"):
            _mem_path.write_text(result + "\n")
            print("  [memory consolidated]", flush=True)

    def _sys_messages():
        sp = load_system_prompt(include_memory=args.memory)
        return [{"role": "system", "content": sp}] if sp else []

    def _wait_for_chime_gap():
        """Wait until we're in a silent gap between ticks, so sd.stop() doesn't
        clip a tick mid-cycle (which clicks). Max wait ~40ms."""
        if chime_sound is None or chime_started_at[0] == 0:
            return
        CHIME_HEAD = 0.22  # end of chime tones in buffer
        TICK_DUR = 0.04    # tick length
        TICK_EVERY = 1.5
        t = _time.monotonic() - chime_started_at[0]
        if t < CHIME_HEAD:
            # Still in chime head; wait for end of chime then it's safe
            _time.sleep(CHIME_HEAD - t)
            return
        phase = (t - CHIME_HEAD) % TICK_EVERY
        if phase < TICK_DUR:
            # In a tick — wait until it ends
            _time.sleep(TICK_DUR - phase + 0.005)

    def play_tts_stream(sentence_source):
        """Play TTS for a sentence source (str or iterator of sentences).

        AEC safety invariants kept across sentence boundaries:
        - Single AEC processor (preserves learned room impulse response)
        - Continuous tts_16k_buf + mic_pos (no alignment drift)
        - Single output stream kept open (no click/pop from teardown)
        - Silence padding in tts_16k_buf for gaps so mic_pos stays aligned
        - 150ms blanking window after each sentence suppresses reverb-tail
          false positives before handing gap mic chunks to AEC
        - Per-sentence inhibit reset: 0.5s protection window applies to every
          sentence start, not just the first one
        """
        if isinstance(sentence_source, str):
            sentence_iter: object = iter(_split_sentences(sentence_source) or [sentence_source])
        else:
            sentence_iter = sentence_source

        drain_audio_q()
        out_stream, interrupted = None, False
        tts_16k_buf: list[np.ndarray] = []
        # Cache for np.concatenate(tts_16k_buf) — recomputed only when list grows.
        _concat_cache: dict = {"arr": np.array([], dtype=np.float32), "len": 0}
        state = {"play_start": None, "consec_speech": 0, "mic_pos": 0}
        aec_process = make_aec_processor() if make_aec_processor else None

        def _get_tts_concat():
            if len(tts_16k_buf) != _concat_cache["len"]:
                _concat_cache["arr"] = np.concatenate(tts_16k_buf) if tts_16k_buf else np.array([], dtype=np.float32)
                _concat_cache["len"] = len(tts_16k_buf)
            return _concat_cache["arr"]

        def _append_ref(chunk_samples, sr):
            if aec_process is None:
                return
            if sr == SAMPLE_RATE:
                tts_16k_buf.append(chunk_samples.astype(np.float32))
            else:
                idx = np.arange(0, len(chunk_samples), sr / SAMPLE_RATE)
                tts_16k_buf.append(
                    np.interp(idx, np.arange(len(chunk_samples)), chunk_samples).astype(np.float32)
                )

        def check_barge_in():
            if not (aec_process and state["play_start"]):
                return False
            if _time.monotonic() - state["play_start"] < 0.5:
                return False
            tts_concat = _get_tts_concat()
            if len(tts_concat) == 0:
                return False
            while not audio_q.empty():
                mic_chunk = audio_q.get_nowait()
                if len(mic_chunk) < CHUNK_SAMPLES:
                    continue
                ref = _get_ref_segment(tts_concat, state["mic_pos"], len(mic_chunk))
                state["mic_pos"] += len(mic_chunk)
                cleaned = aec_process(mic_chunk, ref)
                if _vad_prob(vad, cleaned.astype(np.float32)) > 0.8:
                    state["consec_speech"] += 1
                    if state["consec_speech"] >= 5:
                        return True
                else:
                    state["consec_speech"] = 0
            return False

        def pad_gap_and_check():
            """Drain mic chunks from the inter-sentence gap.

            First _GAP_BLANK_SAMPLES samples: blanked (reverb tail). mic_pos
            advances and silence is appended to tts_16k_buf for alignment, but
            AEC is not called — feeding zero reference during decay would pass
            residual echo through as speech. After blanking, AEC resumes with
            zero reference (silence period is real by then).
            """
            if aec_process is None:
                return False
            blanked = 0
            while not audio_q.empty():
                mic_chunk = audio_q.get_nowait()
                if len(mic_chunk) < CHUNK_SAMPLES:
                    continue
                silence_ref = np.zeros(len(mic_chunk), dtype=np.float32)
                tts_16k_buf.append(silence_ref)  # keep mic_pos aligned
                state["mic_pos"] += len(mic_chunk)
                if blanked < _GAP_BLANK_SAMPLES:
                    state["consec_speech"] = 0
                    blanked += len(mic_chunk)
                    continue
                cleaned = aec_process(mic_chunk, silence_ref)
                if _vad_prob(vad, cleaned.astype(np.float32)) > 0.8:
                    state["consec_speech"] += 1
                    if state["consec_speech"] >= 5:
                        return True
                else:
                    state["consec_speech"] = 0
            return False

        async def _play():
            nonlocal out_stream, interrupted
            first_sentence = True
            for sentence in sentence_iter:
                if interrupted:
                    break
                # Between sentences: handle gap mic chunks before the new TTS
                # stream starts writing to tts_16k_buf so mic_pos is correct.
                if not first_sentence and pad_gap_and_check():
                    interrupted = True
                    print("  [voice interrupt]", flush=True)
                    break
                first_sentence = False
                sentence_first_chunk = True
                tts_sentence_stream = kokoro.create_stream(
                    sentence, voice=args.voice, speed=1.0, lang=_lang_from_voice(args.voice)
                )
                async for chunk_samples, sr in tts_sentence_stream:
                    if out_stream is None:
                        if chime_sound is not None:
                            _wait_for_chime_gap()
                            sd.stop()
                        out_stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                        out_stream.start()
                        drain_audio_q()
                    # Per-sentence inhibit reset: give each sentence its own 0.5s
                    # protection window and a clean consec_speech counter so
                    # residual state from the inter-sentence gap can't accumulate.
                    if sentence_first_chunk:
                        vad.reset_states()
                        state["play_start"] = _time.monotonic()
                        state["consec_speech"] = 0
                        sentence_first_chunk = False
                    _append_ref(chunk_samples, sr)
                    data = chunk_samples.reshape(-1, 1)
                    for i in range(0, len(data), 4096):
                        if select.select([sys.stdin], [], [], 0)[0]:
                            sys.stdin.read(1); interrupted = True
                        elif check_barge_in():
                            interrupted = True; print("  [voice interrupt]", flush=True)
                        if interrupted:
                            break
                        out_stream.write(data[i:i+4096])
                    if interrupted:
                        break
            if out_stream:
                out_stream.stop(); out_stream.close()

        asyncio.run(_play())
        if interrupted and state["consec_speech"] < 3:
            print("  [interrupted]")
        drain_audio_q()
        vad.reset_states()
        return interrupted

    def process_utterance(audio, history):
        print(f" ({len(audio) / SAMPLE_RATE:.1f}s)")
        if chime_sound is not None:
            print("  *chime*", flush=True)
            sd.play(chime_sound, CHIME_SR)
            chime_started_at[0] = _time.monotonic()
        wav_path = save_wav(audio) if args.audio_mode else None
        heard, response = "", ""
        try:
            messages = _sys_messages()
            for h in history[-MAX_HISTORY:]:
                messages += [{"role": "user", "content": h["user"]},
                             {"role": "assistant", "content": h["assistant"]}]
            if args.audio_mode:
                transcribe_future = executor.submit(transcribe, audio)
                messages.append({"role": "user", "content": [{"type": "audio"}]})
            else:
                heard = transcribe(audio)
                print(f"  [{heard}]")
                messages.append({"role": "user", "content": heard})
            if args.audio_mode:
                # Audio mode: no sentence streaming (transcription runs in parallel)
                response = llm_generate(messages, audio=[wav_path])
                heard = transcribe_future.result(timeout=10)
                print(f"  [{heard}]")
                print(f"\n> {response}\n", flush=True)
                if kokoro and response:
                    play_tts_stream(response)
                elif chime_sound is not None:
                    _wait_for_chime_gap()
                    sd.stop()
            else:
                # Text mode: stream sentences — TTS starts on first sentence while
                # LLM continues generating the rest.
                response_parts: list[str] = []

                def _collecting(gen):
                    for s in gen:
                        response_parts.append(s)
                        print(f"> {s}", flush=True)
                        yield s

                print()
                if kokoro:
                    play_tts_stream(_collecting(stream_sentences(messages)))
                else:
                    for _ in _collecting(stream_sentences(messages)):
                        pass
                    if chime_sound is not None:
                        _wait_for_chime_gap()
                        sd.stop()

                response = " ".join(response_parts)
                print()
            history.append({"user": heard, "assistant": response})
            if len(history) > MAX_HISTORY:
                history.pop(0)
            if args.memory:
                update_memory(heard, response)
                if len(history) % 5 == 0:
                    consolidate_memory()
        except Exception as e:
            print(f"\nError: {e}\n", file=sys.stderr)
        finally:
            if wav_path:
                os.unlink(wav_path)

    history, buf = [], []
    chime_started_at = [0.0]  # monotonic time when last chime started (for tick-boundary TTS start)
    speaking, silent_chunks = False, 0

    # Set terminal to raw mode so keypress interrupts work without Enter
    old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    mode = "audio" if args.audio_mode else "text"
    print(f"\nListening (mode: {mode}, tts: {args.tts}, silence: {args.silence_ms}ms, smart-turn: {args.smart_turn})")
    tts_hint = (" Speak or press any key to interrupt TTS." if args.aec else " Press any key to interrupt TTS.") if args.tts else ""
    print(f"Speak into your microphone. Ctrl+C to quit.{tts_hint}\n", flush=True)

    greeting = llm_generate(_sys_messages() + [
        {"role": "user", "content": (
            "Greet the user as Voice Loop in one short sentence. "
            "If my name is in memory, use it and ask how you can help. "
            "Otherwise, ask for my name."
        )},
    ], max_tokens=60)
    print(f"> {greeting}\n", flush=True)
    if kokoro:
        speak_tts(greeting)

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=CHUNK_SAMPLES, callback=callback,
    ):
        try:
            while True:
                chunk = audio_q.get()
                if len(chunk) < CHUNK_SAMPLES:
                    continue

                speech_prob = _vad_prob(vad, chunk)
                if speech_prob > 0.5:
                    if not speaking:
                        speaking = True
                        print("[listening...]", end="", flush=True)
                    silent_chunks = 0
                    buf.append(chunk)
                elif speaking:
                    silent_chunks += 1
                    buf.append(chunk)
                    if silent_chunks < silence_limit:
                        continue
                    if smart_turn and buf:
                        prob = smart_turn(np.concatenate(buf))
                        print(f" [turn prob: {prob:.2f}]", end="", flush=True)
                        if prob < 0.5:
                            silent_chunks = 0
                            continue
                    process_utterance(np.concatenate(buf), history)
                    buf.clear()
                    speaking, silent_chunks = False, 0
                    vad.reset_states()

        except KeyboardInterrupt:
            print("\nBye!")
            executor.shutdown(wait=False)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
            if args.record and record_buf:
                full = np.concatenate(record_buf)
                with wave.open(args.record, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                    wf.writeframes((full * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
                print(f"Recorded {len(full) / SAMPLE_RATE:.1f}s to {args.record}", flush=True)


if __name__ == "__main__":
    main()
