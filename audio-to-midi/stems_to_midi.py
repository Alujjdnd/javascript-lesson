"""Cleaner transcription of the 'other' stem + tidy bass/drums.

Fixes the messy harmony track:
  - stricter basic-pitch thresholds
  - merge same-pitch notes separated by tiny gaps
  - drop short/quiet ghost notes
  - snap starts/durations to a 16th-note grid derived from beat tracking
  - cap simultaneous polyphony, keeping the loudest notes
"""
import numpy as np
import librosa
import pretty_midi
from basic_pitch.inference import predict

TEMPO = 129.2
STEMS = "separated/htdemucs/input_stereo"
OUT = "recreation.mid"

# ---- beat grid from the original audio ----
y, sr = librosa.load("input_mono.wav", sr=22050)
_, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=TEMPO)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
# extend a regular grid from median period, anchored to median beat phase
period = float(np.median(np.diff(beat_times)))
phase = float(np.median((beat_times - np.arange(len(beat_times)) * period)))
grid_step = period / 4.0  # 16th notes
grid = np.arange(phase % grid_step - grid_step, 30.0, grid_step)
grid = grid[grid >= 0]

def snap(t):
    return float(grid[np.argmin(np.abs(grid - t))])

def quantize_notes(notes, min_dur=grid_step * 0.9):
    out = []
    for n in notes:
        s = snap(n.start)
        e = max(s + grid_step, snap(n.end))
        out.append(pretty_midi.Note(n.velocity, n.pitch, s, e))
    return out

def merge_notes(notes, gap=0.09):
    """Merge consecutive same-pitch notes with small gaps."""
    by_pitch = {}
    for n in sorted(notes, key=lambda n: (n.pitch, n.start)):
        by_pitch.setdefault(n.pitch, []).append(n)
    merged = []
    for pitch, ns in by_pitch.items():
        cur = ns[0]
        for n in ns[1:]:
            if n.start - cur.end <= gap:
                cur = pretty_midi.Note(max(cur.velocity, n.velocity), pitch,
                                       cur.start, max(cur.end, n.end))
            else:
                merged.append(cur)
                cur = n
        merged.append(cur)
    return merged

def limit_polyphony(notes, max_poly=4):
    """At any note-on, keep only the loudest max_poly overlapping notes."""
    notes = sorted(notes, key=lambda n: n.start)
    kept = []
    for n in notes:
        overlapping = [k for k in kept if k.end > n.start + 0.03]
        if len(overlapping) >= max_poly:
            weakest = min(overlapping, key=lambda k: k.velocity)
            if weakest.velocity < n.velocity:
                kept.remove(weakest)
                kept.append(n)
        else:
            kept.append(n)
    return sorted(kept, key=lambda n: n.start)

pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)

# ---------- BASS: transcribe, keep strongest line, quantize ----------
_, _, bass_events = predict(
    f"{STEMS}/bass.wav", onset_threshold=0.4, frame_threshold=0.3,
    minimum_note_length=90, maximum_frequency=librosa.midi_to_hz(57))
bass_notes = []
for s, e, p, a, _ in bass_events:
    p = int(p)
    if p > 55:
        continue
    bass_notes.append(pretty_midi.Note(int(np.clip(a * 150, 45, 115)), p, float(s), float(e)))
bass_notes = merge_notes(bass_notes, gap=0.10)
# drop octave-doubled ghosts: if a stronger note an octave below overlaps, drop
strong = []
for n in bass_notes:
    dbl = [m for m in bass_notes if m is not n and (n.pitch - m.pitch) == 12
           and m.start < n.end and n.start < m.end and m.velocity >= n.velocity]
    if not dbl:
        strong.append(n)

# Re-articulate: rhythm comes from the bass stem's own onsets, pitch from
# whichever transcribed note is active there. Repeated same-pitch pulses that
# basic-pitch merges into one long note get their attacks back.
yb, srb = librosa.load(f"{STEMS}/bass.wav", sr=22050, mono=True)
b_onsets = librosa.onset.onset_detect(y=yb, sr=srb, units="time", backtrack=False)
b_strength = librosa.onset.onset_strength(y=yb, sr=srb)
b_times = librosa.frames_to_time(np.arange(len(b_strength)), sr=srb)

def active_pitch(t):
    cands = [n for n in strong if n.start - 0.08 <= t < n.end + 0.05]
    if not cands:
        # nearest note within 0.25s
        near = min(strong, key=lambda n: min(abs(n.start - t), abs(n.end - t)))
        if min(abs(near.start - t), abs(near.end - t)) < 0.25:
            return near.pitch, near.velocity
        return None, None
    best = max(cands, key=lambda n: n.velocity)
    return best.pitch, best.velocity

onset_qs = sorted({snap(t) for t in b_onsets})
bass_notes = []
for i, tq in enumerate(onset_qs):
    p, v = active_pitch(tq)
    if p is None:
        continue
    nxt = onset_qs[i + 1] if i + 1 < len(onset_qs) else tq + 2 * grid_step
    # sustain until the next articulation (or note end), slight release gap
    src_end = max((n.end for n in strong if n.pitch == p and n.start - 0.1 <= tq < n.end + 0.05),
                  default=tq + 2 * grid_step)
    end = min(nxt - 0.02, snap(src_end) + grid_step)
    if end - tq < 0.09:
        end = tq + grid_step
    sv = b_strength[np.argmin(np.abs(b_times - tq))]
    vel = int(np.clip(55 + sv * 6, 50, 118))
    bass_notes.append(pretty_midi.Note(vel, p, float(tq), float(end)))
bass = pretty_midi.Instrument(program=33, name="Bass")
bass.notes = bass_notes
pm.instruments.append(bass)
print(f"Bass: {len(bass_notes)} notes")

# ---------- HARMONY: strict thresholds + heavy cleanup ----------
_, _, harm_events = predict(
    f"{STEMS}/other.wav", onset_threshold=0.65, frame_threshold=0.45,
    minimum_note_length=140,
    minimum_frequency=librosa.midi_to_hz(43))
harm_notes = []
for s, e, p, a, _ in harm_events:
    p = int(p)
    dur = e - s
    if a < 0.25 or dur < 0.12 or p < 43 or p > 88:
        continue
    harm_notes.append(pretty_midi.Note(int(np.clip(a * 140, 40, 110)), p, float(s), float(e)))
harm_notes = merge_notes(harm_notes, gap=0.12)
harm_notes = quantize_notes(harm_notes)
harm_notes = merge_notes(harm_notes, gap=0.02)   # re-merge after snapping
harm_notes = limit_polyphony(harm_notes, max_poly=4)
harm = pretty_midi.Instrument(program=89, name="Harmony/Pad")  # warm pad
harm.notes = harm_notes
pm.instruments.append(harm)
print(f"Harmony: {len(harm_notes)} notes")

# ---------- DRUMS: as before but quantized ----------
import scipy.signal as ss
yd, srd = librosa.load(f"{STEMS}/drums.wav", sr=22050, mono=True)
onsets = librosa.onset.onset_detect(y=yd, sr=srd, units="time", backtrack=False, delta=0.03)
strengths = librosa.onset.onset_strength(y=yd, sr=srd)
otimes = librosa.frames_to_time(np.arange(len(strengths)), sr=srd)

def band(y_, lo, hi, t):
    sos = ss.butter(4, [lo, hi], "bp", fs=srd, output="sos")
    yf = ss.sosfilt(sos, y_)
    i0 = int(t * srd)
    seg = yf[i0:i0 + int(0.05 * srd)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0

drum = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
seen = set()
for t in onsets:
    tq = snap(t)
    kick, snare, hat = band(yd, 30, 120, t), band(yd, 150, 800, t), band(yd, 5000, 10000, t)
    total = kick + snare + hat + 1e-9
    vel = int(np.clip(60 + strengths[np.argmin(np.abs(otimes - t))] * 8, 50, 120))
    hits = []
    if kick / total > 0.35: hits.append((36, vel))
    if snare / total > 0.35: hits.append((38, vel))
    if hat / total > 0.20: hits.append((42, max(40, vel - 20)))
    if not hits: hits = [(42, vel)]
    for p, v in hits:
        if (p, round(tq, 3)) in seen:
            continue
        seen.add((p, round(tq, 3)))
        drum.notes.append(pretty_midi.Note(v, p, tq, tq + (0.1 if p != 42 else 0.05)))
pm.instruments.append(drum)
print(f"Drums: {len(drum.notes)} hits")

pm.write(OUT)
print("wrote", OUT)
