"""Continuation v2: the loop plays through again, intact.

The recording cuts off mid-bar-16 exactly as the cycle returns to its Ab
downbeat, so the continuation is a whole-block copy of bars 1-15 pasted at
bar 16 — every phrase, sustain, and drum pattern in its original relationship,
at the original energy. Only subtle humanization is applied (velocity +-3,
pitched timing +-6 ms) so the second pass isn't a bit-exact repeat. The
source's few partial notes after the bar-16 line are removed (the paste
replaces them); sustains that started earlier tail over the seam and glue it.
A single resolving Bb-minor bar closes the piece.
"""
import numpy as np
import pretty_midi

rng = np.random.default_rng(7)
BAR = 0.4644 * 4
FIRST = 0.08127
SEAM = FIRST + 15 * BAR          # 27.95 s — bar 16 downbeat, loop restart
BLOCK = 15 * BAR                 # length of the copied cycle
OUTRO = FIRST + 30 * BAR         # 55.81 s — final chord downbeat

src = pretty_midi.PrettyMIDI('recreation_v42n.mid')
out = pretty_midi.PrettyMIDI(initial_tempo=129.2)

for inst in src.instruments:
    ni = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
    # first pass: keep everything before the seam; drop the partial bar-16
    # onsets (the paste re-creates that bar in full)
    for n in inst.notes:
        if n.start < SEAM:
            ni.notes.append(pretty_midi.Note(n.velocity, n.pitch, n.start, n.end))
    # second pass: block-copy bars 1-15, shifted one full cycle
    for n in inst.notes:
        if not (FIRST - 0.15 <= n.start < SEAM):
            continue
        if inst.is_drum:
            s, e, v = n.start + BLOCK, n.end + BLOCK, n.velocity
        else:
            jit = float(rng.normal(0, 0.006))
            s = n.start + BLOCK + jit
            e = n.end + BLOCK + jit
            v = int(np.clip(n.velocity + rng.normal(0, 3), 40, 118))
        e = min(e, OUTRO + 0.25)          # nothing rings past the final chord
        if e - s > 0.03:
            ni.notes.append(pretty_midi.Note(v, n.pitch, s, e))
    out.instruments.append(ni)

by = {i.name: i for i in out.instruments}
# closing bar: Bb minor, voiced like the song's own Bbm bars, full sustain
L = 4.4
by['Bass'].notes.append(pretty_midi.Note(84, 34, OUTRO, OUTRO + L))                 # Bb1
for p, v in [(46, 72), (53, 70), (58, 66)]:                                         # Bb2 F3 Bb3
    by['Acoustic Piano'].notes.append(pretty_midi.Note(v, p, OUTRO + 0.01, OUTRO + L))
for p, v in [(58, 62), (61, 58), (65, 60), (70, 55)]:                               # Bb3 Db4 F4 Bb4
    by['Strings'].notes.append(pretty_midi.Note(v, p, OUTRO + 0.03, OUTRO + L + 0.8))
by['Drums'].notes.append(pretty_midi.Note(92, 36, OUTRO, OUTRO + 0.1))              # final kick
by['Drums'].notes.append(pretty_midi.Note(60, 49, OUTRO, OUTRO + 0.1))              # crash wash

out.write('song_extended.mid')
total = sum(len(i.notes) for i in out.instruments)
end = max(n.end for i in out.instruments for n in i.notes)
print(f"{total} notes, ends {end:.2f}s")
# per-bar density check: second pass should mirror the first
def density(a, b):
    return sum(1 for i in out.instruments for n in i.notes if a <= n.start < b)
for k in range(15):
    d1 = density(FIRST + k * BAR, FIRST + (k + 1) * BAR)
    d2 = density(SEAM + k * BAR, SEAM + (k + 1) * BAR)
    print(f"bar {k+1:>2}: pass1={d1:>3} pass2={d2:>3}")
