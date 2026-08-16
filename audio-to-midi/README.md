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

## MIDI tracks

1. **Bass** (Fingered Electric Bass, program 33) — 42 notes. Rhythm is taken
   from the bass stem's own detected onsets (all 42 covered) and pitch from
   the basic-pitch transcription, so repeated same-pitch pulses keep their
   attacks. Moves through A♭–C♯/D♭–E♭–F–B♭ territory, fitting B♭ minor.
2. **Harmony/Pad** (Warm Pad, program 89) — 106 notes. Strict transcription
   thresholds plus post-processing: same-pitch note merging, ghost-note
   removal, 16th-note grid quantization, and a 4-voice polyphony cap.
3. **Drums** (GM channel 10) — 109 hits classified into kick (36), snare (38),
   and closed hi-hat (42) by band-energy analysis, quantized to the grid.

All pitched notes (148/148) fall inside the B♭ natural minor scale.

## Method

1. Decode MP3 → WAV (ffmpeg).
2. Global analysis with librosa: tempo/beat tracking, Krumhansl key
   estimation, HPSS, structure segmentation.
3. Source separation with **Demucs (htdemucs)** → bass / drums / other /
   vocals stems.
4. Pitched stems transcribed with **Spotify basic-pitch** (per-stem pitch
   ranges and onset/frame thresholds), note velocities from model amplitude.
5. Drum stem: onset detection + band-energy classification (30–120 Hz kick,
   150–800 Hz snare, 5–10 kHz hats), layered hits allowed.
6. All tracks merged into one MIDI at 129.2 BPM with pretty_midi.

## Validation (rendered MIDI vs. original)

- Chroma cosine similarity: **0.89 mean** (harmonic content matches closely)
- Tempo of rendered audio: **129.2 BPM** (identical)
- **94 %** of the original's detected onsets matched within 70 ms
- Bass rhythm: **42/42** of the bass stem's onsets have a MIDI note
