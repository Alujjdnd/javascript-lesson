"""v6: best single-model transcription (v4/v42n) refined by the ensemble.

- VERIFY: every v42n note gets a cross-family support count and CQT evidence
  score. Notes with zero cross-family support AND weak evidence are pruned.
- FILL: family-voted consensus clusters with strong evidence that v42n
  missed entirely are added (duration from a YourMT3+ member when present,
  role by nearest-in-register v42n track, velocity from evidence).
- Drums likewise: v42n hits verified against the 2-family drum consensus
  (weak-energy unsupported hits pruned), consensus hits absent from v42n
  added.
"""
import json
import numpy as np
import librosa
import pretty_midi

ONSET_TOL = 0.06
y, sr = librosa.load('input_mono.wav', sr=22050)
C = np.abs(librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'),
                       n_bins=96, bins_per_octave=12))
def ev(pitch, t0, t1):
    b = pitch - 24
    if not (0 <= b < 96): return 0.0
    f0 = max(0, int(t0*sr/512)); f1 = max(f0+1, int(min(t1, t0+0.3)*sr/512))
    return float(C[b, f0:f1].max())

def notes_from_midi(path, drums=False):
    pm = pretty_midi.PrettyMIDI(path)
    out = []
    for inst in pm.instruments:
        if inst.is_drum != drums: continue
        for n in inst.notes:
            out.append(dict(s=n.start, e=n.end, p=n.pitch, v=n.velocity, tr=inst.name))
    return out

def notes_from_json(path):
    return [dict(s=s, e=e, p=p, tr=None) for s, e, p in json.load(open(path))]

FAMS = {
 'ymt3': (notes_from_midi('recreation_v3.mid') +
          notes_from_midi('ymt3space/model_output/stem_bass.mid') +
          notes_from_midi('ymt3space/model_output/stem_other.mid')),
 'bp':   (notes_from_midi('fullmix_basicpitch.mid') +
          notes_from_json('candidates/bp_htdemucs_bass.json') +
          notes_from_json('candidates/bp_htdemucs_other.json') +
          notes_from_json('candidates/bp_ftstems_bass.json') +
          notes_from_json('candidates/bp_ftstems_other.json')),
 'pyin': (notes_from_json('candidates/pyin_bass_htdemucs.json') +
          notes_from_json('candidates/pyin_bass_ft.json')),
}

def support_of(p, s, exclude_fam=None):
    fams = set()
    for fam, notes in FAMS.items():
        if fam == exclude_fam: continue
        for n in notes:
            if n['p'] == p and abs(n['s'] - s) <= ONSET_TOL:
                fams.add(fam); break
    return fams

base = pretty_midi.PrettyMIDI('recreation_v42n.mid')
ev_samples = [ev(n.pitch, n.start, n.end) for i in base.instruments
              if not i.is_drum for n in i.notes]
EV_STRONG = np.percentile(ev_samples, 55)
EV_WEAK = np.percentile(ev_samples, 8)

out = pretty_midi.PrettyMIDI(initial_tempo=129.2)
pruned = 0
kept_notes = []       # (pitch, start) for dedup when filling
for inst in base.instruments:
    ni = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
    out.instruments.append(ni)
    if inst.is_drum:
        continue
    for n in inst.notes:
        # v42n derives from ymt3; support must come from other families
        sup = support_of(n.pitch, n.start, exclude_fam='ymt3')
        e_ = ev(n.pitch, n.start, n.end)
        if not sup and e_ < EV_WEAK:
            pruned += 1
            continue
        ni.notes.append(pretty_midi.Note(n.velocity, n.pitch, n.start, n.end))
        kept_notes.append((n.pitch, n.start))

# ---- FILL: cross-family clusters v42n missed ----
added = 0
by_reg = sorted([i for i in out.instruments if not i.is_drum],
                key=lambda i: np.median([n.pitch for n in i.notes]) if i.notes else 0)
def nearest_track(pitch):
    best, bd = None, 1e9
    for i in out.instruments:
        if i.is_drum or not i.notes: continue
        d = abs(np.median([n.pitch for n in i.notes]) - pitch)
        if d < bd: best, bd = i, d
    return best

# candidate fills: bp∩pyin or bp∩ymt3-stem notes not already in v42n
pool = []
for fam in ('bp', 'pyin'):
    for n in FAMS[fam]:
        pool.append((n, fam))
seen = set()
for n, fam in pool:
    key = (n['p'], round(n['s'] / ONSET_TOL))
    if key in seen: continue
    seen.add(key)
    if any(p == n['p'] and abs(s - n['s']) <= ONSET_TOL for p, s in kept_notes):
        continue
    fams = support_of(n['p'], n['s'])
    if len(fams) < 2:
        continue
    e_ = ev(n['p'], n['s'], n['e'])
    if e_ < EV_STRONG:
        continue
    # duration: prefer a ymt3 member's duration if one matched
    dur = n['e'] - n['s']
    for m in FAMS['ymt3']:
        if m['p'] == n['p'] and abs(m['s'] - n['s']) <= ONSET_TOL:
            dur = m['e'] - m['s']; break
    dur = max(0.1, min(dur, 3.0))
    tr = nearest_track(n['p'])
    r = (np.log1p(e_) - np.log1p(EV_WEAK)) / max(1e-9, np.log1p(np.percentile(ev_samples, 97)) - np.log1p(EV_WEAK))
    vel = int(np.clip(45 + 70 * r, 42, 112))
    tr.notes.append(pretty_midi.Note(vel, n['p'], n['s'], n['s'] + dur))
    kept_notes.append((n['p'], n['s']))
    added += 1

# ---- drums ----
DRUMMAP = {35:'kick',36:'kick',38:'snare',39:'snare',40:'snare',42:'hat',44:'hat',46:'hat',
           41:'tom',43:'tom',45:'tom',47:'tom',48:'tom',50:'tom',
           49:'cym',51:'cym',52:'cym',53:'cym',55:'cym',57:'cym'}
CANON = {'kick':36,'snare':38,'hat':42,'tom':45,'cym':49}
DF = {
 'ymt3': ([ (n['s'], DRUMMAP.get(n['p'])) for n in notes_from_midi('recreation_v3.mid', drums=True)] +
          [ (n['s'], DRUMMAP.get(n['p'])) for n in notes_from_midi('ymt3space/model_output/stem_drums.mid', drums=True)]),
 'bands':([ (s, DRUMMAP.get(p)) for s,e,p in json.load(open('candidates/drums_bands_htdemucs.json'))] +
          [ (s, DRUMMAP.get(p)) for s,e,p in json.load(open('candidates/drums_bands_ft.json'))]),
}
def dsupport(cls, s, exclude=None):
    fams = set()
    for fam, hits in DF.items():
        if fam == exclude: continue
        if any(c == cls and abs(t - s) <= ONSET_TOL for t, c in hits):
            fams.add(fam)
    return fams

yd, _ = librosa.load('separated/htdemucs/input_stereo/drums.wav', sr=22050)
d_onsets = librosa.onset.onset_detect(y=yd, sr=sr, units='time', backtrack=False, delta=0.02)
ddrum = [i for i in out.instruments if i.is_drum][0]
basedrum = [i for i in base.instruments if i.is_drum][0]
dpruned = dadded = 0
existing = []
for n in basedrum.notes:
    cls = DRUMMAP.get(n.pitch)
    sup = dsupport(cls, n.start, exclude='ymt3')
    near_onset = len(d_onsets) and np.min(np.abs(d_onsets - n.start)) < 0.05
    if not sup and not near_onset:
        dpruned += 1; continue
    ddrum.notes.append(pretty_midi.Note(n.velocity, n.pitch, n.start, n.end))
    existing.append((cls, n.start))
for s, cls in DF['bands']:
    if cls is None: continue
    if any(c == cls and abs(t - s) <= ONSET_TOL for c, t in existing):
        continue
    if len(dsupport(cls, s)) < 2:
        continue
    ddrum.notes.append(pretty_midi.Note(80, CANON[cls], s, s + (0.05 if cls in ('hat','cym') else 0.1)))
    existing.append((cls, s))
    dadded += 1

# hygiene
for inst in out.instruments:
    inst.notes.sort(key=lambda n: (n.pitch, n.start))
    keep = []
    for n in inst.notes:
        if keep and keep[-1].pitch == n.pitch and n.start < keep[-1].end:
            keep[-1].end = n.start
            if keep[-1].end - keep[-1].start < 0.04: keep.pop()
        keep.append(n)
    inst.notes = sorted(keep, key=lambda n: n.start)

out.write('recreation_v6.mid')
print(f"pitched: pruned {pruned}, added {added}")
print(f"drums: pruned {dpruned}, added {dadded}")
for i in out.instruments:
    print(f"  {i.name:<16} {len(i.notes)}")
