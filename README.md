# Voice Loop

A minimal on-device voice agent loop. Runs entirely on Mac M4 / Apple Silicon.

> Need a custom voice model or production voice agent? See [Trelis Voice AI Services](https://trelis.com/voice-ai-services/).

## Features

- **Smart turn detection** — Silero VAD + pipecat's Smart Turn v3, so the agent waits when you pause mid-sentence
- **Voice interruption** — speak over the agent; WebRTC AEC3 cancels echo from speakers so your voice cuts through
- **Editable persona** — `SOUL.md` controls the agent's style, live-reloaded each turn
- **Optional long-term memory** — enable with `--memory`; the agent learns durable facts about you in `MEMORY.md` and consolidates every 5 turns
- **Fully local** — no API keys, no cloud. Everything runs on-device

## Stack

- **Moonshine** (CPU) for speech-to-text transcription
- **Gemma 4 E4B** (MLX/Metal) for response generation
- **Kokoro** (CPU) for TTS (streaming)
- **Silero VAD** + **Smart Turn v3** for turn detection
- **WebRTC AEC3** (via LiveKit APM) for voice interruption

## Setup

```bash
brew install portaudio espeak-ng
git clone https://github.com/TrelisResearch/voice-loop.git
cd voice-loop
uv sync
```

First run downloads Gemma 4 E4B (~3GB), Moonshine (~250MB), Kokoro (~300MB).

## Usage

```bash
# Recommended defaults (TTS + smart turn + voice interrupt all on)
uv run voice_loop_mac.py

# + chime and soft ticks while generating
uv run voice_loop_mac.py --chime-loop

# + persistent memory (reads/writes MEMORY.md)
uv run voice_loop_mac.py --memory

# Text-only mode (no TTS)
uv run voice_loop_mac.py --no-tts

# Disable voice interruption (keypress only)
uv run voice_loop_mac.py --no-aec

# Different voice (see below)
uv run voice_loop_mac.py --voice bf_emma

# Use the smaller E2B model (faster, slightly lower quality)
uv run voice_loop_mac.py --model mlx-community/gemma-4-E2B-it-4bit

# Custom silence timeout
uv run voice_loop_mac.py --silence-ms 500

# Debug: record mic stream to a WAV
uv run voice_loop_mac.py --record
```

## Recommended Kokoro voices

Only the higher-quality voices are listed here:

| Voice | Accent | Gender | Notes |
|-------|--------|--------|-------|
| `af_heart` | US | Female | **Top pick** — Grade A (default) |
| `af_bella` | US | Female | Grade A-, HH training |
| `bf_emma` | UK | Female | Grade B-, HH training |
| `am_fenrir` | US | Male | Grade C+, H training |
| `am_puck` | US | Male | Grade C+, H training |
| `am_michael` | US | Male | Grade C+, H training |
| `bm_fable` | UK | Male | Grade C, MM training |
| `bm_george` | UK | Male | Grade C, MM training |

## Architecture

```
   Mic (16kHz) ──► Silero VAD ──► Smart Turn ──► Moonshine ──► Gemma 4 E4B ──► Kokoro ──► Speakers
                                                                    ▲                         │
                                                        SOUL.md + MEMORY.md                   │
                                                                                              ▼
   Mic during TTS ──► WebRTC AEC3 (LiveKit APM) ──► Silero VAD ──► voice interrupt ◄──────────┘
```

## How it works

1. **Mic capture** via sounddevice (16kHz mono)
2. **Silero VAD** detects speech vs silence
3. **Smart Turn** confirms end-of-turn on silence (default on)
4. **Moonshine** transcribes your audio to text (CPU)
5. **Gemma 4 E4B** responds using SOUL.md (+ MEMORY.md if `--memory`) as system prompt
6. **Kokoro** synthesizes speech, streams audio
7. **WebRTC AEC3** cleans mic during TTS playback → Silero VAD on cleaned audio → voice interrupt

Press any key during TTS to interrupt.

## Persona & Memory

- `SOUL.md` — persona / style (always loaded, live-reloaded each turn)
- `MEMORY.md` — long-term facts. Only read/written when `--memory` is passed. When enabled, the agent extracts new durable facts after each turn and consolidates every 5 turns.

Both files are re-read at the start of every turn, so edits take effect immediately.

## Memory usage

~3.5 GB total. Fits easily in 16GB.

## Credits

Built with:
- [Moonshine](https://github.com/moonshine-ai/moonshine) — STT
- [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) — TTS
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) — end-of-turn detection
- [LiveKit APM](https://github.com/livekit/python-sdks) — WebRTC AEC3
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — MLX multimodal inference
- [Gemma 4](https://huggingface.co/google/gemma-4-E4B-it) — LLM

## License

Apache 2.0.
