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
import select
import sys
import tempfile
import termios
import time as _time
import tty
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz (required by Silero VAD)
MAX_HISTORY = 10
CHIME_SR = 24000
_DIR = Path(__file__).parent


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

def make_chime():
    # Single short chime ack only (played on its own by --chime flag)
    return np.concatenate([_fade_tone(880, 0.09), _silence(0.03), _fade_tone(1320, 0.10)])

def make_chime_with_ticks(duration=30.0, tick_every=1.5, tick_start=1.2):
    """Chime + silence + periodic soft ticks, all in one buffer — one sd.play() call."""
    parts = [_fade_tone(880, 0.09), _silence(0.03), _fade_tone(1320, 0.10)]
    head = np.concatenate(parts)
    tick = _fade_tone(660, 0.08, amp=0.35)
    total = int(duration * CHIME_SR)
    buf = np.zeros(total, dtype=np.float32)
    buf[:len(head)] = head
    step = int(tick_every * CHIME_SR)
    for pos in range(int(tick_start * CHIME_SR), total, step):
        end = min(pos + len(tick), total)
        buf[pos:end] = tick[:end - pos]
    return buf

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
    ap.add_argument("--chime", action="store_true", help="Play chime when utterance detected")
    ap.add_argument("--chime-loop", action="store_true", help="Chime + soft ticks while generating")
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
    chime_loop_sound = make_chime_with_ticks() if args.chime_loop else None
    chime_sound = make_chime() if (args.chime and not args.chime_loop) else None
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

    def speak_tts(text):
        samples, sr = kokoro.create(text, voice=args.voice, speed=1.0, lang="en-us")
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

    def play_tts_stream(response):
        drain_audio_q()
        tts_stream = kokoro.create_stream(response, voice=args.voice, speed=1.0, lang="en-us")
        out_stream, interrupted = None, False
        tts_16k_buf: list[np.ndarray] = []
        state = {"play_start": None, "consec_speech": 0, "mic_pos": 0}
        aec_process = make_aec_processor() if make_aec_processor else None

        def check_barge_in():
            if not (aec_process and state["play_start"] and tts_16k_buf):
                return False
            if _time.monotonic() - state["play_start"] < 0.5:
                return False
            tts_concat = np.concatenate(tts_16k_buf)
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

        async def _play():
            nonlocal out_stream, interrupted
            async for chunk_samples, sr in tts_stream:
                if out_stream is None:
                    if chime_loop_sound is not None:
                        sd.stop()
                    out_stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                    out_stream.start()
                    drain_audio_q(); vad.reset_states()
                    state["play_start"] = _time.monotonic()
                if aec_process is not None:
                    if sr == SAMPLE_RATE:
                        tts_16k_buf.append(chunk_samples.astype(np.float32))
                    else:
                        idx = np.arange(0, len(chunk_samples), sr / SAMPLE_RATE)
                        tts_16k_buf.append(np.interp(idx, np.arange(len(chunk_samples)), chunk_samples).astype(np.float32))
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
        # Start chime + tick loop in one call (kills device switch click)
        if chime_loop_sound is not None:
            print("  *chime*", flush=True)
            sd.play(chime_loop_sound, CHIME_SR)
        elif chime_sound is not None:
            print("  *chime*", flush=True)
            sd.play(chime_sound, CHIME_SR); sd.wait()
        wav_path = save_wav(audio) if args.audio_mode else None
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
            response = llm_generate(messages, **({"audio": [wav_path]} if args.audio_mode else {}))
            if args.audio_mode:
                heard = transcribe_future.result(timeout=10)
                print(f"  [{heard}]")
            print(f"\n> {response}\n", flush=True)
            if kokoro and response:
                play_tts_stream(response)
            elif chime_loop_sound is not None:
                sd.stop()
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
