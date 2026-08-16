"""Score MIDI transcriptions for stray notes and artifacts.

Metrics per file:
  - notes, tracks/programs (instrument leakage shows as many sparse programs)
  - fragment rate: notes shorter than 100 ms
  - out-of-key rate vs Bb natural minor (+ raised 7th A as borderline)
  - octave ghosts: overlapping note exactly +12 semitones at lower velocity
  - same-pitch overlaps (stuck/duplicated notes)
  - pitch outliers: > 12 semitones from the track's rolling neighborhood
  - off-grid: median onset deviation from the 16th-note grid (129.2 BPM)
"""
import sys
import numpy as np
import pretty_midi

KEY = {10, 0, 1, 3, 5, 6, 8}      # Bb natural minor
KEY_SOFT = KEY | {9}              # + raised 7th (harmonic minor)
STEP, FIRST = 0.1161, 0.08127     # 16th grid from the pipeline

def analyze(path):
    pm = pretty_midi.PrettyMIDI(path)
    rows = []
    tot = dict(notes=0, frag=0, offkey=0, offkey_hard=0, ghosts=0, overlaps=0, outliers=0, devs=[])
    for inst in pm.instruments:
        ns = sorted(inst.notes, key=lambda n: n.start)
        if not ns:
            continue
        n_notes = len(ns)
        frag = sum(1 for n in ns if n.end - n.start < 0.1)
        if inst.is_drum:
            offkey = offkey_hard = ghosts = outliers = 0
        else:
            offkey = sum(1 for n in ns if n.pitch % 12 not in KEY_SOFT)
            offkey_hard = sum(1 for n in ns if n.pitch % 12 not in KEY)
            ghosts = 0
            for i, n in enumerate(ns):
                for m in ns:
                    if m.pitch == n.pitch - 12 and m.start < n.end and n.start < m.end \
                       and m.velocity > n.velocity:
                        ghosts += 1
                        break
            outliers = 0
            for n in ns:
                near = [m.pitch for m in ns if m is not n and abs(m.start - n.start) < 0.75]
                if near and min(abs(n.pitch - p) for p in near) > 12:
                    outliers += 1
        overlaps = 0
        for i, n in enumerate(ns):
            for m in ns[i + 1:]:
                if m.start >= n.end:
                    break
                if m.pitch == n.pitch:
                    overlaps += 1
        devs = [min(abs(((n.start - FIRST) % STEP)), STEP - ((n.start - FIRST) % STEP)) for n in ns]
        name = inst.name or f"prog{inst.program}"
        rows.append((name, inst.is_drum, n_notes, frag, offkey, ghosts, overlaps, outliers,
                     1000 * float(np.median(devs))))
        tot['notes'] += n_notes; tot['frag'] += frag; tot['offkey'] += offkey
        tot['offkey_hard'] += offkey_hard
        tot['ghosts'] += ghosts; tot['overlaps'] += overlaps; tot['outliers'] += outliers
        tot['devs'] += devs
    return rows, tot

for path in sys.argv[1:]:
    rows, t = analyze(path)
    print(f"\n=== {path} ===")
    print(f"{'track':<28}{'notes':>6}{'<100ms':>8}{'offkey':>8}{'ghost':>7}{'ovlp':>6}{'outlr':>7}{'grid-ms':>9}")
    for r in rows:
        drum = ' [dr]' if r[1] else ''
        print(f"{(r[0]+drum)[:27]:<28}{r[2]:>6}{r[3]:>8}{'-' if r[1] else r[4]:>8}"
              f"{'-' if r[1] else r[5]:>7}{r[6]:>6}{'-' if r[1] else r[7]:>7}{r[8]:>9.1f}")
    n = max(1, t['notes'])
    print(f"TOTAL {t['notes']} notes | fragments {t['frag']} ({100*t['frag']/n:.0f}%) | "
          f"off-key {t['offkey']} soft / {t['offkey_hard']} hard ({100*t['offkey']/n:.0f}%) | "
          f"ghosts {t['ghosts']} | same-pitch overlaps {t['overlaps']} | outliers {t['outliers']} | "
          f"median grid dev {1000*np.median(t['devs']):.0f} ms")
