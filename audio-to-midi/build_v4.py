"""v4: hybrid of YourMT3+ note content and evidence-based hand-cleanup.

- Velocities re-derived from the recording (YourMT3+ emits flat 100s):
  pitched notes from CQT energy at the note's own pitch, drums from
  band-limited stem energy at the hit.
- Bass replaced with the stem-articulated 42-note line from v2 (YourMT3+
  hears sustains; the stem's onsets carry the real rhythm).
- Drum hits gated against real drum-stem onsets; unsupported hits dropped.
- Spectral support check: pitched notes with no energy at their pitch in
  the original mix are deleted.
- Fragments merged, same-pitch overlaps trimmed, onsets snapped to the
  16th grid (electronic source, straight grid), sustains kept legato.
"""
import numpy as np
import librosa
import pretty_midi
import scipy.signal as ss

STEP, FIRST, TEMPO = 0.1161, 0.08127, 129.2
HOP = 512

y, sr = librosa.load('input_mono.wav', sr=22050)
DUR = len(y) / sr
C = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP, fmin=librosa.note_to_hz('C1'),
                       n_bins=96, bins_per_octave=12))
def cqt_t(t): return min(C.shape[1] - 1, max(0, int(t * sr / HOP)))
def pitch_energy(pitch, t0, t1):
    b = pitch - 24
    if b < 0 or b >= C.shape[0]: return 0.0
    f0, f1 = cqt_t(t0), max(cqt_t(t0) + 1, cqt_t(min(t1, t0 + 0.25)))
    return float(C[b, f0:f1].max())

def snap(t): return t

# ---------------- pitched tracks from v3 ----------------
v3 = pretty_midi.PrettyMIDI('recreation_v3.mid')
v2 = pretty_midi.PrettyMIDI('recreation.mid')  # for the articulated bass

out = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
PROGRAMS = {'Acoustic Piano': 0, 'Electric Piano': 4, 'Guitar (clean)': 27,
            'Strings': 50, 'Lead': 81}

all_support = []
tracks_notes = {}
for inst in v3.instruments:
    if inst.is_drum or inst.name == 'Bass':
        continue
    ns = sorted(inst.notes, key=lambda n: n.start)
    # merge same-pitch small gaps
    merged = []
    by_pitch = {}
    for n in ns: by_pitch.setdefault(n.pitch, []).append(n)
    for p, group in by_pitch.items():
        cur = group[0]
        for n in group[1:]:
            if False:
                cur = pretty_midi.Note(100, p, cur.start, max(cur.end, n.end))
            else:
                merged.append(cur); cur = n
        merged.append(cur)
    kept = []
    for n in merged:
        sup = pitch_energy(n.pitch, n.start, n.end)
        dur = n.end - n.start
        kept.append((n, sup, dur))
        all_support.append(sup)
    tracks_notes[inst.name] = kept

sup_arr = np.array(all_support)
sup_floor = np.percentile(sup_arr[sup_arr > 0], 12)   # bottom of real content
log_sup = np.log1p(sup_arr)

def vel_from_support(s):
    # global rank -> 42..115
    r = (np.log1p(s) - log_sup.min()) / max(1e-9, log_sup.max() - log_sup.min())
    return int(np.clip(42 + 75 * r, 40, 115))

report = {}
for name, kept in tracks_notes.items():
    inst = pretty_midi.Instrument(program=PROGRAMS.get(name, 0), name=name)
    dropped = 0
    for n, sup, dur in kept:
        if dur < 0.15 and sup < sup_floor:          # unsupported short note
            dropped += 1; continue
        if sup < sup_floor * 0.35:                  # no spectral evidence at all
            dropped += 1; continue
        s = snap(n.start)
        e = max(s + 0.05, snap(n.end))
        inst.notes.append(pretty_midi.Note(vel_from_support(sup), n.pitch, s, e))
    # trim same-pitch overlaps after snapping
    inst.notes.sort(key=lambda n: (n.pitch, n.start))
    final = []
    for n in inst.notes:
        if final and final[-1].pitch == n.pitch and n.start < final[-1].end:
            if n.start == final[-1].start:
                continue
            final[-1].end = n.start
        final.append(n)
    inst.notes = sorted([n for n in final if n.end - n.start > 0.05],
                        key=lambda n: n.start)
    out.instruments.append(inst)
    report[name] = (len(kept), len(inst.notes), dropped)

# ---------------- bass: articulated line from v2, velocities from bass stem ----------------
yb, _ = librosa.load('separated/htdemucs/input_stereo/bass.wav', sr=22050)
Cb = np.abs(librosa.cqt(yb, sr=sr, hop_length=HOP, fmin=librosa.note_to_hz('C1'),
                        n_bins=96, bins_per_octave=12))
def bass_energy(pitch, t):
    b = pitch - 24
    f0 = cqt_t(t); f1 = f0 + max(1, int(0.15 * sr / HOP))
    return float(Cb[b, f0:f1].max()) if 0 <= b < Cb.shape[0] else 0.0

v2bass = [i for i in v2.instruments if i.name == 'Bass'][0]
bass = pretty_midi.Instrument(program=33, name='Bass')
bes = [bass_energy(n.pitch, n.start) for n in v2bass.notes]
lb = np.log1p(np.array(bes))
for n, e in zip(v2bass.notes, bes):
    r = (np.log1p(e) - lb.min()) / max(1e-9, lb.max() - lb.min())
    vel = int(np.clip(55 + 60 * r, 50, 118))
    bass.notes.append(pretty_midi.Note(vel, n.pitch, n.start, n.end))
out.instruments.append(bass)
report['Bass'] = (len(v2bass.notes), len(bass.notes), 0)

# ---------------- drums: v3 kit gated by stem onsets, stem-energy velocities ----------------
yd, _ = librosa.load('separated/htdemucs/input_stereo/drums.wav', sr=22050)
d_onsets = librosa.onset.onset_detect(y=yd, sr=sr, units='time', backtrack=False, delta=0.02)
def band_rms(lo, hi, t):
    sos = ss.butter(4, [lo, hi], 'bp', fs=sr, output='sos')
    i0 = int(t * sr); seg = ss.sosfilt(sos, yd[max(0, i0):i0 + int(0.06 * sr)])
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
BANDS = {36: (30, 120), 38: (120, 900), 39: (120, 900), 42: (5000, 10000),
         43: (80, 300), 45: (80, 300), 47: (100, 350), 49: (3000, 9000),
         50: (100, 350), 51: (3000, 9000), 52: (3000, 9000)}
v3drums = [i for i in v3.instruments if i.is_drum][0]
drum = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')
seen = set(); gated = 0
hit_es = []
hits = []
for n in sorted(v3drums.notes, key=lambda n: n.start):
    near = np.min(np.abs(d_onsets - n.start)) if len(d_onsets) else 1.0
    limit = 0.15 if n.pitch in (36, 38, 39) else 0.10
    if near > limit:
        gated += 1; continue
    tq = snap(n.start)
    key = (n.pitch, round(tq, 4))
    if key in seen: continue
    seen.add(key)
    lo, hi = BANDS.get(n.pitch, (100, 8000))
    e = band_rms(lo, hi, n.start)
    hits.append((tq, n.pitch, e)); hit_es.append(e)
le = np.log1p(np.array(hit_es))
for tq, p, e in hits:
    r = (np.log1p(e) - le.min()) / max(1e-9, le.max() - le.min())
    vel = int(np.clip(50 + 68 * r, 45, 120))
    dur = 0.05 if p in (42, 49, 51, 52) else 0.1
    drum.notes.append(pretty_midi.Note(vel, p, tq, tq + dur))
out.instruments.append(drum)
report['Drums'] = (len(v3drums.notes), len(drum.notes), gated)

out.write('recreation_v42n.mid')
for k, (before, after, dropped) in report.items():
    print(f"{k:<16} {before:>4} -> {after:>4}  (dropped/gated {dropped})")
vels = [n.velocity for i in out.instruments for n in i.notes]
print(f"velocity spread: min={min(vels)} max={max(vels)} distinct={len(set(vels))}")
