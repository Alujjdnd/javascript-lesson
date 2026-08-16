"""Consensus merge: vote across independent transcription systems.

A note survives if (a) >=2 independent systems heard it (same pitch, onsets
within tolerance), or (b) exactly one system heard it but the original
recording's CQT shows strong harmonic evidence at that pitch and time.
Merged onset/offset are medians over the agreeing systems. Velocities come
from CQT energy (duration-agnostic). Instruments follow YourMT3+ labels when
available, register otherwise. Drums vote in a reduced kick/snare/hat/tom/cym
vocabulary across four drum systems.
"""
import json
import numpy as np
import librosa
import pretty_midi

ONSET_TOL = 0.06
y, sr = librosa.load('input_mono.wav', sr=22050)
C = np.abs(librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'),
                       n_bins=96, bins_per_octave=12))

def cqt_evidence(pitch, t0, t1):
    """Max CQT magnitude at the pitch bin over the note head, normalized later."""
    b = pitch - 24
    if not (0 <= b < 96): return 0.0
    f0 = max(0, int(t0 * sr / 512)); f1 = max(f0 + 1, int(min(t1, t0 + 0.3) * sr / 512))
    return float(C[b, f0:f1].max())

def notes_from_midi(path, drums=False):
    pm = pretty_midi.PrettyMIDI(path)
    out = []
    for inst in pm.instruments:
        if inst.is_drum != drums: continue
        for n in inst.notes:
            out.append((n.start, n.end, n.pitch, inst.name))
    return out

def notes_from_json(path):
    return [(s, e, p, None) for s, e, p in json.load(open(path))]

# ---------------- pitched systems ----------------
SYSTEMS = {
    'ymt3_fullmix': notes_from_midi('recreation_v3.mid', drums=False),
    'ymt3_stems':  (notes_from_midi('ymt3space/model_output/stem_bass.mid') +
                    notes_from_midi('ymt3space/model_output/stem_other.mid')),
    'bp_fullmix':   notes_from_midi('fullmix_basicpitch.mid'),
    'bp_htdemucs': (notes_from_json('candidates/bp_htdemucs_bass.json') +
                    notes_from_json('candidates/bp_htdemucs_other.json')),
    'bp_ftstems':  (notes_from_json('candidates/bp_ftstems_bass.json') +
                    notes_from_json('candidates/bp_ftstems_other.json')),
    'pyin_bass':   (notes_from_json('candidates/pyin_bass_htdemucs.json') +
                    notes_from_json('candidates/pyin_bass_ft.json')),
}

def cluster(systems, tol=ONSET_TOL):
    """Greedy clustering of notes across systems by (pitch, onset)."""
    pool = []
    for sysname, notes in systems.items():
        for s, e, p, label in notes:
            pool.append(dict(s=s, e=e, p=int(p), sys=sysname, label=label))
    pool.sort(key=lambda n: (n['p'], n['s']))
    clusters = []
    cur = None
    for n in pool:
        if cur and n['p'] == cur['p'] and n['s'] - cur['last_s'] <= tol:
            cur['members'].append(n); cur['last_s'] = n['s']
        else:
            cur = dict(p=n['p'], members=[n], last_s=n['s'])
            clusters.append(cur)
    return clusters

clusters = cluster(SYSTEMS)

# evidence normalization: distribution over all cluster heads
ev_all = np.array([cqt_evidence(c['p'], min(m['s'] for m in c['members']),
                                max(m['e'] for m in c['members'])) for c in clusters])
ev_pos = ev_all[ev_all > 0]
EV_STRONG = np.percentile(ev_pos, 60)   # bar for single-system survival
EV_MIN = np.percentile(ev_pos, 15)      # bar for everything

FAMILY = {'ymt3_fullmix': 'ymt3', 'ymt3_stems': 'ymt3',
          'bp_fullmix': 'bp', 'bp_htdemucs': 'bp', 'bp_ftstems': 'bp',
          'pyin_bass': 'pyin'}
merged = []
for c, ev in zip(clusters, ev_all):
    systems_in = {m['sys'] for m in c['members']}
    families_in = {FAMILY[s] for s in systems_in}
    support = len(families_in)          # count architectures, not variants
    if support < 2 and not (ev >= EV_STRONG and 'ymt3' in families_in):
        continue
    if ev < EV_MIN:
        continue
    onsets = sorted(m['s'] for m in c['members'])
    offsets = sorted(m['e'] for m in c['members'])
    s = float(np.median(onsets)); e = float(np.median(offsets))
    if e - s < 0.05: e = s + 0.05
    labels = [m['label'] for m in c['members'] if m['label']]
    label = max(set(labels), key=labels.count) if labels else None
    merged.append(dict(s=s, e=e, p=c['p'], support=support, ev=float(ev), label=label))

print(f"pitched: {len(clusters)} clusters -> {len(merged)} kept")
from collections import Counter
print("support histogram:", dict(sorted(Counter(m['support'] for m in merged).items())))

# ---------------- drums ----------------
DRUMMAP = {35:'kick',36:'kick',38:'snare',39:'snare',40:'snare',
           42:'hat',44:'hat',46:'hat',41:'tom',43:'tom',45:'tom',47:'tom',48:'tom',50:'tom',
           49:'cym',51:'cym',52:'cym',53:'cym',55:'cym',57:'cym'}
CANON = {'kick':36,'snare':38,'hat':42,'tom':45,'cym':49}
DSYS = {
    'ymt3_fullmix': notes_from_midi('recreation_v3.mid', drums=True),
    'ymt3_stem':    notes_from_midi('ymt3space/model_output/stem_drums.mid', drums=True),
    'bands_htd':    notes_from_json('candidates/drums_bands_htdemucs.json'),
    'bands_ft':     notes_from_json('candidates/drums_bands_ft.json'),
}
dpool = {}
for name, notes in DSYS.items():
    dpool[name] = [(s, DRUMMAP.get(p)) for s, e, p, _ in notes if DRUMMAP.get(p)]

dclusters = []
allhits = sorted(((s, cls, name) for name, hits in dpool.items() for s, cls in hits),
                 key=lambda x: (x[1], x[0]))
cur = None
for s, cls, name in allhits:
    if cur and cls == cur['cls'] and s - cur['last'] <= ONSET_TOL:
        cur['members'].append((s, name)); cur['last'] = s
    else:
        cur = dict(cls=cls, members=[(s, name)], last=s)
        dclusters.append(cur)

DFAMILY = {'ymt3_fullmix': 'ymt3', 'ymt3_stem': 'ymt3',
           'bands_htd': 'bands', 'bands_ft': 'bands'}
dmerged = []
for c in dclusters:
    systems_in = {n for _, n in c['members']}
    families_in = {DFAMILY[n] for n in systems_in}
    if len(families_in) < 2:
        continue
    s = float(np.median([t for t, _ in c['members']]))
    dmerged.append(dict(s=s, cls=c['cls'], support=len(systems_in)))
print(f"drums: {len(dclusters)} clusters -> {len(dmerged)} kept (>=2 families)")
print("drum class counts:", dict(Counter(m['cls'] for m in dmerged)))

# ---------------- assemble MIDI ----------------
PROG = {'Acoustic Piano': ('Acoustic Piano', 0), 'Electric Piano': ('Electric Piano', 4),
        'Guitar (clean)': ('Guitar (clean)', 27), 'Strings': ('Strings', 50),
        'Singing Voice': ('Lead', 81), 'Lead': ('Lead', 81), 'Bass': ('Bass', 33)}
def role_of(m):
    if m['label'] in PROG: return PROG[m['label']][0]
    return 'Bass' if m['p'] < 45 else ('Strings' if m['e'] - m['s'] > 0.6 else 'Electric Piano')

pm = pretty_midi.PrettyMIDI(initial_tempo=129.2)
insts = {}
lo, hi = np.log1p(EV_MIN), np.log1p(np.percentile(ev_pos, 97))
for m in merged:
    role = role_of(m)
    if role not in insts:
        prog = dict(PROG.values())[role] if role in dict(PROG.values()) else 0
        prog = {v[0]: v[1] for v in PROG.values()}[role]
        insts[role] = pretty_midi.Instrument(program=prog, name=role)
        pm.instruments.append(insts[role])
    r = (np.log1p(m['ev']) - lo) / max(1e-9, hi - lo)
    vel = int(np.clip(45 + 70 * r, 42, 115))
    insts[role].notes.append(pretty_midi.Note(vel, m['p'], m['s'], m['e']))

drum = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')
yd, _ = librosa.load('separated/htdemucs/input_stereo/drums.wav', sr=22050)
import scipy.signal as ss
BANDS = {'kick': (30,120), 'snare': (120,900), 'hat': (5000,10000),
         'tom': (80,300), 'cym': (3000,9000)}
es = []
for m in dmerged:
    lo_, hi_ = BANDS[m['cls']]
    sos = ss.butter(4, [lo_, hi_], 'bp', fs=sr, output='sos')
    i0 = int(m['s']*sr); seg = ss.sosfilt(sos, yd[i0:i0+int(0.06*sr)])
    es.append(float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0)
le = np.log1p(np.array(es))
for m, e in zip(dmerged, es):
    r = (np.log1p(e) - le.min()) / max(1e-9, le.max() - le.min())
    vel = int(np.clip(50 + 65 * r, 45, 118))
    dur = 0.05 if m['cls'] in ('hat','cym') else 0.1
    drum.notes.append(pretty_midi.Note(vel, CANON[m['cls']], m['s'], m['s'] + dur))
pm.instruments.append(drum)

# same-pitch overlap hygiene
for inst in pm.instruments:
    inst.notes.sort(key=lambda n: (n.pitch, n.start))
    keep = []
    for n in inst.notes:
        if keep and keep[-1].pitch == n.pitch and n.start < keep[-1].end:
            keep[-1].end = n.start
            if keep[-1].end - keep[-1].start < 0.04: keep.pop()
        keep.append(n)
    inst.notes = sorted(keep, key=lambda n: n.start)

pm.write('recreation_v5.mid')
for i in pm.instruments:
    print(f"  {i.name:<16} {len(i.notes)} notes")
print("wrote recreation_v5.mid")
