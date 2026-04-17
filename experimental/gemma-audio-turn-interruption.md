# Gemma Audio for Turn Detection & Interruption Handling

## Motivation

Current pipeline: `VAD → TD model → STT → Gemma (text)`

The TD model adds latency and has no semantic understanding — it can't distinguish "wait let me think about..." (mid-turn pause) from a genuine end-of-turn. Idea: collapse the pipeline using Gemma's audio input directly.

---

## Benchmark Results (M4 Pro, MLX, 4-bit)

Tested `mlx-community/gemma-4-E2B-it-4bit` and `E4B` with synthetic audio.
Model already cached locally — load time ~3s. Audio feature extractor: 16kHz, 8s chunks, 750 audio tokens/chunk.

| Model | Window | p50 latency | Fits 250ms poll? |
|-------|--------|-------------|------------------|
| E2B   | 5.0s   | 308ms       | No               |
| E2B   | 7.5s   | 363ms       | No               |
| E2B   | 10.0s  | 453ms       | No               |
| E4B   | 5.0s   | 567ms       | No               |
| E4B   | 10.0s  | 813ms       | No               |

**Practical cadence:** E2B with a 5s rolling window, polled every ~350ms is viable.
E4B is too slow for real-time polling on this hardware.

### Caveat on synthetic audio
The benchmark used sine-wave audio (150Hz + harmonics). The model outputs valid labels (`MID_PAUSE`, `SPEAKING`) but this only validates latency — the model may be responding to the prompt text as much as the audio content. Real validation requires labelled real-speech samples (complete sentence / mid-pause / mid-word cutoff).

---

## Proposed Architecture

### 1. Turn detection — collapse TD model into Gemma

**Current:** `VAD → TD model → STT → Gemma`

**Proposed:** `VAD → Gemma (audio mode)`

Gemma starts generating immediately after VAD fires. First token signals intent:
- Normal response token → stream to TTS as usual
- `[CONTINUE]` token → discard, wait for more speech, append to context

Benefits:
- Eliminates the TD model entirely
- Gemma's semantic + prosodic understanding strictly better than TD model
- Knows "wait let me think..." is not end-of-turn
- Can predict turn end before silence arrives (sees trajectory of utterance)

Cost: STT (Moonshine) still needed for history/display — run as parallel speculative call.

---

### 2. Interruption handling — backchannel classification

**Current:** `VAD fires during TTS → stop immediately`

**Proposed:** `VAD fires → pause TTS → Gemma classifies → REAL_INTERRUPT or BACKCHANNEL`

#### On BACKCHANNEL — hold + resume state machine

```
VAD fires during TTS
  → pause TTS immediately
  → suppress VAD for 150ms  (AEC settling window)
  → Gemma classifies

  BACKCHANNEL:
    backchannel_count += 1
    if backchannel_count >= 2 → treat as REAL_INTERRUPT
    else → hold 400ms
      → silence for 400ms → resume from last sentence boundary
      → VAD fires again (after 150ms settle) → restart classification

  REAL_INTERRUPT:
    yield floor, discard TTS buffer
```

Key details:
- **Resume from last sentence boundary** (not mid-stream) — keeps AEC reference clean and avoids choppy mid-word restart
- **150ms VAD suppression** after pause — lets AEC re-stabilise before listening again, otherwise speaker echo tail triggers VAD immediately
- **Escalation after 2 backchannels** — handles "yeah yeah I know" without infinite loops; no recursion

#### Why not resume mid-stream?
AEC reference gets out of sync on mid-buffer resume — echo bleedthrough at the resume point can re-trigger VAD and create a feedback loop.

---

## Data Requirements (if fine-tuning)

To fine-tune Gemma for this task:

| Label | Examples needed |
|-------|----------------|
| `END_TURN` | Complete sentences with falling intonation, trailing silence |
| `MID_PAUSE` | Filled pauses ("um", "uh"), hesitation mid-sentence |
| `SPEAKING` | Active mid-utterance, cut off mid-word |
| `BACKCHANNEL` | "mm-hmm", "yeah", "right", "uh-huh" over assistant speech |
| `REAL_INTERRUPT` | User starts new sentence over assistant |

Estimated: 10–50 hours annotated conversation audio for fine-tuning a pre-trained model.

**Permissive datasets:**
- AMI Meeting Corpus — CC BY 4.0, annotated turns
- DailyTalk — CC BY-SA, dyadic dialogue
- ICSI Meeting Corpus — publicly available

(Fisher, Switchboard, CallHome are LDC-licensed / paid — avoid.)

---

## Human Turn-Taking Reference

- Average inter-turn gap in natural conversation: **~200ms** (Stivers et al., 2009 — cross-linguistic)
- Humans start planning their turn **~600–800ms before** the current speaker finishes — they predict, don't react
- This is why predictive turn detection (seeing the prosodic trajectory) beats silence-based VAD

---

## Open Questions

- Does Gemma audio actually outperform the current TD model on real speech? Needs benchmark with real labelled samples.
- E2B at 350ms polling cadence: acceptable UX or still too slow? Compare against current TD model latency.
- `[CONTINUE]` token: add as a literal special token to fine-tune on, or use a text prefix the system detects?
- For backchannel classification: run Gemma on the AEC output (post-echo-cancel) or raw mic? AEC output is cleaner signal.
