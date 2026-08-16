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

## MIDI tracks (v3 — YourMT3+)

`recreation.mid` is now transcribed by **YourMT3+** (YPTF.MoE checkpoint,
MLSP 2024) reading the full mix in one pass: Acoustic Piano (45), Electric
Piano (45), Guitar (69), Bass (21), Strings (101), Lead (6), Drums (138 hits,
full kit incl. toms/cymbals). All 287 pitched notes are in B♭ minor with zero
octave-ghost artifacts. The earlier Demucs + basic-pitch version is kept as
`recreation_basicpitch.mid`; `analyze_midi.py` scores any MIDI for stray-note
artifacts, and `yourmt3_transcribe.py` is the model runner.

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

## Validation (rendered MIDI vs. original)

| metric | v1 basic-pitch | v2 cleaned | v3 YourMT3+ |
|---|---|---|---|
| octave ghosts | 141 | 10 | **0** |
| off-key notes | 0 | 0 | **0** |
| onset match (70 ms) | 62 % | 94 % | **96 %** |
| chroma similarity | 0.95 | 0.89 | 0.88 |
| tracks | 3 | 3 | **7** |

A stem-wise YourMT3+ run (Demucs stems transcribed separately) was also
tested and scored slightly worse than the full-mix pass on both stray-note
and audio metrics.
