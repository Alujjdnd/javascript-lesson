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

## Method (v3)

1. Decode MP3 → WAV (ffmpeg); tempo/key analysis with librosa
   (129.2 BPM, B♭ minor via Krumhansl profile).
2. Transcribe the full mix with **YourMT3+** (YPTF.MoE+Multi noPS checkpoint
   from the official HF space), CPU inference at fp32.
3. Control experiment: Demucs (htdemucs) stems transcribed separately with
   the same model — scored worse, not adopted.
4. Light cleanup: the model's six-note "Singing Voice" figure reassigned to
   a synth-lead program; timing left unquantized.

Earlier versions: v1/v2 used Demucs stems + Spotify basic-pitch with
band-energy drum classification and grid quantization (see
`stems_to_midi.py`, kept for reference).

## Continuation

`song_extended.mid` continues the piece from bar 16 (where the recording cuts
off mid-bar) with a 10-bar self-sampled coda (`continue_song.py`): the chord
cycle (A♭ → D♭ → Fm → B♭m → Fm → G♭ → Fm → B♭m → E♭m) is derived from per-bar
analysis, each new bar borrows phrases from source bars with the same chord
(velocity-shaped, ±8 ms jitter), and the arc eases down to a sustained B♭ minor
resolution (~53 s total). Continuation vs source: chroma profile correlation
0.926, identical tempo, 144/144 notes in key, gently softer dynamics.
Preview: `song_extended_preview.mp3`.

## Validation (rendered MIDI vs. original)

| metric | v1 basic-pitch | v2 cleaned | v3 YourMT3+ | v4 hand-finished |
|---|---|---|---|---|
| off-key notes | 0 | 0 | 0 | **0** |
| MIDI onset coverage (70 ms) | — | — | 47/47 | **47/47** |
| chroma similarity | 0.95* | 0.89 | 0.88 | **0.91** |
| velocity levels | varied | varied | 1 (flat 100) | **70** |
| tracks | 3 | 3 | 7 | **7** |

\* v1's higher chroma came from 141 phantom octave-ghost notes padding the
spectrum — not accuracy.

A stem-wise YourMT3+ run (Demucs stems transcribed separately) was also
tested and scored slightly worse than the full-mix pass on both stray-note
and audio metrics.
