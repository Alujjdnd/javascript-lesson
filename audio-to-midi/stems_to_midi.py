"""Transcribe demucs stems into one multi-track MIDI file.

Tracks:
  - bass   -> basic-pitch (monophonic-ish, low register), Electric Bass
  - vocals -> basic-pitch, Voice-ish lead (if stem has energy)
  - other  -> basic-pitch, harmony/keys
  - drums  -> band-energy onset classification (kick/snare/hat) on GM ch10
"""
import sys
import numpy as np
import librosa
import pretty_midi
import scipy.signal as ss

STEM_DIR = sys.argv[1] if len(sys.argv) > 1 else "separated/htdemucs/input_stereo"
OUT = sys.argv[2] if len(sys.argv) > 2 else "recreation.mid"
TEMPO = float(sys.argv[3]) if len(sys.argv) > 3 else 129.2

from basic_pitch.inference import predict

pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)


def stem_rms(path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    return float(np.sqrt(np.mean(y ** 2))), y, sr


def add_pitched_track(path, name, program, min_pitch=0, max_pitch=127,
                      onset_thresh=0.5, frame_thresh=0.3, min_note_len=80):
    _, midi_data, note_events = predict(
        path,
        onset_threshold=onset_thresh,
        frame_threshold=frame_thresh,
        minimum_note_length=min_note_len,
        minimum_frequency=librosa.midi_to_hz(min_pitch) if min_pitch else None,
        maximum_frequency=librosa.midi_to_hz(max_pitch) if max_pitch < 127 else None,
    )
    inst = pretty_midi.Instrument(program=program, name=name)
    kept = 0
    for start, end, pitch, amp, _bends in note_events:
        pitch = int(pitch)
        if not (min_pitch <= pitch <= max_pitch):
            continue
        vel = int(np.clip(amp * 127, 30, 120))
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch,
                                           start=float(start), end=float(end)))
        kept += 1
    if kept:
        pm.instruments.append(inst)
    print(f"{name}: {kept} notes")
    return kept


def band_energy(y, sr, lo, hi, t):
    sos = ss.butter(4, [lo, hi], "bp", fs=sr, output="sos")
    yf = ss.sosfilt(sos, y)
    n = int(0.05 * sr)
    i0 = max(0, int(t * sr))
    seg = yf[i0:i0 + n]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


def add_drum_track(path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    if np.sqrt(np.mean(y ** 2)) < 1e-3:
        print("drums: stem silent, skipped")
        return
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time",
                                        backtrack=False, delta=0.03)
    strengths = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.frames_to_time(np.arange(len(strengths)), sr=sr)
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for t in onsets:
        kick = band_energy(y, sr, 30, 120, t)
        snare = band_energy(y, sr, 150, 800, t)
        hat = band_energy(y, sr, 5000, 10000, t)
        total = kick + snare + hat + 1e-9
        idx = np.argmin(np.abs(times - t))
        vel = int(np.clip(60 + strengths[idx] * 8, 50, 120))
        # choose loudest bands; allow layered hits (kick+hat etc.)
        if kick / total > 0.35:
            drum.notes.append(pretty_midi.Note(vel, 36, t, t + 0.1))
        if snare / total > 0.35:
            drum.notes.append(pretty_midi.Note(vel, 38, t, t + 0.1))
        if hat / total > 0.20:
            drum.notes.append(pretty_midi.Note(max(40, vel - 20), 42, t, t + 0.05))
        if not any(abs(n.start - t) < 1e-6 for n in drum.notes):
            drum.notes.append(pretty_midi.Note(vel, 42, t, t + 0.05))
    pm.instruments.append(drum)
    print(f"drums: {len(drum.notes)} hits from {len(onsets)} onsets")


import os

bass_p = os.path.join(STEM_DIR, "bass.wav")
voc_p = os.path.join(STEM_DIR, "vocals.wav")
oth_p = os.path.join(STEM_DIR, "other.wav")

for p, nm in [(bass_p, "bass"), (voc_p, "vocals"), (oth_p, "other")]:
    r, _, _ = stem_rms(p)
    print(f"stem {nm} rms={r:.4f}")

r_voc, _, _ = stem_rms(voc_p)

add_pitched_track(bass_p, "Bass", program=33, min_pitch=24, max_pitch=60,
                  onset_thresh=0.4, frame_thresh=0.3, min_note_len=90)
if r_voc > 0.005:
    add_pitched_track(voc_p, "Lead/Vocal", program=54, min_pitch=48, max_pitch=96,
                      onset_thresh=0.5, frame_thresh=0.3, min_note_len=100)
add_pitched_track(oth_p, "Harmony/Keys", program=81, min_pitch=36, max_pitch=100,
                  onset_thresh=0.5, frame_thresh=0.3, min_note_len=80)
add_drum_track(os.path.join(STEM_DIR, "drums.wav"))

pm.write(OUT)
print("wrote", OUT)
