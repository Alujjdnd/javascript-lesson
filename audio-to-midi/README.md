# Audio → MIDI Recreation

Multi-track MIDI transcription of `source.mp3` (28.6 s instrumental).

## Files

| File | Description |
|---|---|
| `source.mp3` | Original audio |
| `recreation.mid` | The multi-track MIDI recreation (main deliverable) |
| `midi_preview.mp3` | The MIDI rendered back to audio (FluidR3 GM soundfont) for easy A/B comparison |
| `stems_to_midi.py` | Pipeline script that produced the MIDI from separated stems |

## Musical analysis

- **Duration:** 28.6 s
- **Tempo:** ~129 BPM (4/4)
- **Key:** B♭ minor (chroma profile dominated by A♭/G♯, F, B♭, D♭/C♯, E♭/D♯)
- **Instrumentation:** instrumental — bass, harmonic/pad layer, drums. The
  vocal stem separated by Demucs was silent (RMS ≈ 0.0005), so no vocal track.

## MIDI tracks (v4 — YourMT3+ hand-finished)

`recreation.mid` = YourMT3+ note content plus evidence-based cleanup
(`build_v4.py`): velocities re-derived from CQT energy (the raw model emits
flat 100s), bass re-articulated from stem onsets (42 notes), drum hits gated
against drum-stem onsets (128 kept of 138), spectrally unsupported notes
deleted, natural timing kept. Tracks: Acoustic Piano (45), Electric Piano
(44), Guitar (69), Bass (42), Strings (95), Lead (6), Drums (128). All 301
pitched notes in B♭ minor. Raw model output kept as `recreation_ymt3_raw.mid`,
first-generation version as `recreation_basicpitch.mid`; `analyze_midi.py`
scores any MIDI for stray-note artifacts.

## Method (v6 — ensemble-verified pipeline)

Six transcription systems across three architecture families feed a
verification layer (`pipeline/`):

1. **Candidates** (`gen_candidates.py`): YourMT3+ (full mix + htdemucs stems),
   basic-pitch (full mix + htdemucs stems + htdemucs_ft stems), deterministic
   pyin bass tracking; four drum systems (YourMT3+ ×2, band-energy ×2).
2. **Cross-family voting** (`consensus.py`): notes clustered across systems
   (same pitch, onsets ≤60 ms); support counted per architecture family —
   same-family variants vote as one (three basic-pitch variants otherwise
   confirm each other's shared hallucinations).
3. **Verify-and-fill** (`build_v6.py`): the best single-model transcription
   (v4) is the base; notes with no cross-family support and weak CQT evidence
   are pruned (19), cross-family notes it missed with strong evidence are
   added (33); drums verified against stem onsets and the 2-family drum vote.
4. **Scoring** (`eval_framework.py`, mir_eval): note-level F1 between systems,
   bidirectional onset P/R/F, band envelope correlation, tempo-invariant
   rhythm autocorrelation. Rejected as unstable: note counts, duration stats,
   grid deviation, standalone chroma.

Earlier versions: v1/v2 used Demucs stems + Spotify basic-pitch with
band-energy drum classification and grid quantization (see
`stems_to_midi.py`, kept for reference).

## Continuation

`song_extended.mid` (~61 s): the recording cuts off mid-bar exactly as its
15-bar cycle (A♭ → D♭ → Fm → B♭m → Fm → G♭ → Fm → B♭m → E♭m) returns to its
A♭ downbeat, so the continuation block-copies bars 1–15 onto the seam as one
piece (`continue_song.py`) — every phrase and drum pattern in its original
relationship, at the original energy (per-bar note density identical, RMS
within 2 %, envelope corr 0.81, chroma similarity 0.972) — with only ±3
velocity / ±6 ms humanization, closing on a sustained B♭ minor bar with a
kick-and-crash. Preview: `song_extended_preview.mp3`.

## Validation (rendered MIDI vs. original)

| metric (render vs original) | v3 YourMT3+ | v4 hand-finished | v5 pure ensemble | v6 verified |
|---|---|---|---|---|
| onset precision | 0.587 | 0.776 | 0.605 | **0.800** |
| onset recall | 0.936 | 0.809 | 0.979 | **0.851** |
| onset F1 | 0.721 | 0.792 | 0.748 | **0.825** |
| rhythm autocorr similarity | 0.996 | 0.996 | 0.982 | **0.998** |
| off-key notes | 0 | 0 | 0 | **0** |

The pure ensemble (v5) lost to the single best model — consensus works as a
*verifier*, not a *generator*. v6 = v4 base ± ensemble corrections wins on
both precision and recall.

A stem-wise YourMT3+ run (Demucs stems transcribed separately) was also
tested and scored slightly worse than the full-mix pass on both stray-note
and audio metrics.
