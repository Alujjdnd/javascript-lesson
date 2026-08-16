"""Continue the song from bar 16 (where the recording cuts off) through a
10-bar coda, by self-sampling: each new bar copies real phrases from source
bars carrying the same chord, with arrangement thinning and velocity shaping
for a soothing arc, ending on a long resolved Bbm.
"""
import numpy as np
import pretty_midi

rng = np.random.default_rng(11)
BAR = 0.4644 * 4
FIRST = 0.08127
src = pretty_midi.PrettyMIDI('recreation_v42n.mid')

def bar_t(b):            # bar index (0-based) -> start time
    return FIRST + b * BAR

# chord -> source bars (0-based) that carry it, chosen for cleanliness
CHORD_BARS = {
    'Ab':  [0],
    'Db':  [1, 2],
    'Fm':  [3, 4, 7, 8, 11],
    'Bbm': [5, 6, 12],
    'Gb':  [9, 10],
    'Ebm': [13, 14],
}

# continuation plan: (bar_index, chord, tracks-mask, velocity-scale)
# masks: which instrument names play. 'drums-lite' = kick+hat only.
ALL = {'Acoustic Piano', 'Electric Piano', 'Guitar (clean)', 'Strings', 'Bass', 'Drums'}
PLAN = [
    # complete bar 16 (already half-sounding in the source) + settle
    (16, 'Db',  {'Acoustic Piano', 'Strings', 'Bass', 'drums-lite'},          0.82),
    (17, 'Db',  {'Acoustic Piano', 'Electric Piano', 'Strings', 'Bass', 'Drums'}, 0.88),
    (18, 'Fm',  {'Acoustic Piano', 'Electric Piano', 'Strings', 'Bass', 'Drums'}, 0.92),
    (19, 'Fm',  ALL,                                                           0.95),
    (20, 'Bbm', ALL,                                                           1.00),
    (21, 'Bbm', ALL,                                                           0.97),
    (22, 'Gb',  {'Acoustic Piano', 'Strings', 'Guitar (clean)', 'Bass', 'Drums'}, 0.90),
    (23, 'Ebm', {'Acoustic Piano', 'Strings', 'Bass', 'Drums'},               0.85),
    (24, 'Fm',  {'Acoustic Piano', 'Strings', 'Bass', 'drums-lite'},          0.78),
    (25, 'Ab',  {'Strings', 'Bass', 'drums-lite'},                            0.70),
]
OUTRO_BAR = 26   # long Bbm sustain

out = pretty_midi.PrettyMIDI(initial_tempo=129.2)
insts = {}
for i in src.instruments:
    ni = pretty_midi.Instrument(program=i.program, is_drum=i.is_drum, name=i.name)
    ni.notes = [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in i.notes]
    out.instruments.append(ni)
    insts[i.name] = ni

def sample_bar(src_bar, dst_bar, track, vscale, drums_lite=False, keep=1.0):
    t0, t1 = bar_t(src_bar), bar_t(src_bar + 1)
    shift = bar_t(dst_bar) - t0
    added = 0
    for n in insts[track].notes[:]:
        if not (t0 <= n.start < t1):
            continue
        if n.start >= bar_t(16):     # never sample from the continuation itself
            continue
        if drums_lite and track == 'Drums' and n.pitch not in (36, 42):
            continue
        if keep < 1.0 and rng.random() > keep:
            continue
        jit = 0.0 if track == 'Drums' else float(rng.normal(0, 0.008))
        s = n.start + shift + jit
        e = min(n.end + shift + jit, bar_t(dst_bar + 1) + 0.6)
        vel = int(np.clip(n.velocity * vscale + rng.normal(0, 3), 40, 115))
        insts[track].notes.append(pretty_midi.Note(vel, n.pitch, s, e))
        added += 1
    return added

for dst, chord, mask, vscale in PLAN:
    src_bar = int(rng.choice(CHORD_BARS[chord]))
    drums_lite = 'drums-lite' in mask
    for track in ['Acoustic Piano', 'Electric Piano', 'Guitar (clean)', 'Strings', 'Bass', 'Drums']:
        if track in mask or (track == 'Drums' and drums_lite):
            # thin the busiest tracks slightly in quiet bars
            keep = 0.85 if (vscale < 0.85 and track in ('Strings',)) else 1.0
            sample_bar(src_bar, dst, track, vscale, drums_lite=drums_lite, keep=keep)

# ---- outro: long resolved Bbm, soft ----
t0 = bar_t(OUTRO_BAR)
LEN = 4.2
for pitch, vel in [(34, 62)]:                                   # Bb1 bass
    insts['Bass'].notes.append(pretty_midi.Note(vel, pitch, t0, t0 + LEN))
for pitch, vel in [(46, 55), (49, 52), (53, 54), (58, 50)]:     # Bbm keys voicing
    insts['Acoustic Piano'].notes.append(pretty_midi.Note(vel, pitch, t0 + 0.02, t0 + LEN))
for pitch, vel in [(58, 48), (61, 46), (65, 47)]:               # strings above
    insts['Strings'].notes.append(pretty_midi.Note(vel, pitch, t0 + 0.05, t0 + LEN + 0.6))
insts['Lead'].notes.append(pretty_midi.Note(44, 77, t0 + BAR * 0.5, t0 + LEN))  # F5 whisper
insts['Drums'].notes.append(pretty_midi.Note(70, 36, t0, t0 + 0.1))             # final soft kick

out.write('song_extended.mid')
total = sum(len(i.notes) for i in out.instruments)
end = max(n.end for i in out.instruments for n in i.notes)
print(f"extended: {total} notes, ends at {end:.2f}s ({(end-FIRST)/BAR:.1f} bars)")
for i in out.instruments:
    cont = sum(1 for n in i.notes if n.start >= bar_t(16))
    print(f"  {i.name:<16} total {len(i.notes):>3}  continuation {cont:>3}")
