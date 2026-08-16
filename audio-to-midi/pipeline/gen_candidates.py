"""Generate the transcription candidate pool for consensus voting.

Systems (independent by model and/or separation):
  A ymt3_fullmix      YourMT3+ on the full mix          (exists: recreation_v3.mid)
  B ymt3_stems        YourMT3+ on htdemucs stems        (exists: ymt3space/model_output/)
  C bp_fullmix        basic-pitch on the full mix       (exists: fullmix_basicpitch.mid)
  D bp_htdemucs       basic-pitch on htdemucs stems     (generated here)
  E bp_ftstems        basic-pitch on htdemucs_ft stems  (generated here, needs ft stems)
  F pyin_bass         deterministic pyin f0 tracking on both bass stems (generated here)
Drum systems:
  ymt3 drums (A/B), band-energy classifier on both drum stems (generated here)
"""
import os, sys, json
import numpy as np
import librosa

OUT = 'candidates'
os.makedirs(OUT, exist_ok=True)

def note_list_to_json(notes, path):
    json.dump([[round(s,4), round(e,4), int(p)] for s,e,p in notes], open(path,'w'))

# ---------- basic-pitch on a stem set ----------
def bp_stems(stem_dir, tag):
    from basic_pitch.inference import predict
    for stem, opts in [('bass', dict(onset_threshold=0.4, frame_threshold=0.3,
                                     minimum_note_length=90,
                                     maximum_frequency=librosa.midi_to_hz(60))),
                       ('other', dict(onset_threshold=0.5, frame_threshold=0.3,
                                      minimum_note_length=80))]:
        p = os.path.join(stem_dir, stem + '.wav')
        _, _, ev = predict(p, **opts)
        notes = [(float(s), float(e), int(pi)) for s, e, pi, a, _ in ev]
        note_list_to_json(notes, f'{OUT}/{tag}_{stem}.json')
        print(f'{tag}_{stem}: {len(notes)} notes')

# ---------- pyin bass (deterministic) ----------
def pyin_bass(bass_wav, tag):
    y, sr = librosa.load(bass_wav, sr=22050)
    f0, vfl, vpr = librosa.pyin(y, fmin=30, fmax=350, sr=sr, frame_length=4096)
    t = librosa.times_like(f0, sr=sr, hop_length=512)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=False)
    notes = []
    for i, on in enumerate(onsets):
        end = onsets[i+1] if i+1 < len(onsets) else on + 0.8
        sel = (t >= on + 0.02) & (t < min(end, on + 0.5))
        seg = f0[sel]; seg = seg[~np.isnan(seg)]
        if len(seg) < 3: continue
        pitch = int(round(float(np.median(librosa.hz_to_midi(seg)))))
        notes.append((float(on), float(end - 0.02), pitch))
    note_list_to_json(notes, f'{OUT}/{tag}.json')
    print(f'{tag}: {len(notes)} notes')

# ---------- band-energy drum classifier ----------
def drums_bands(drum_wav, tag):
    import scipy.signal as ss
    y, sr = librosa.load(drum_wav, sr=22050)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=False, delta=0.03)
    def band(lo, hi, tt):
        sos = ss.butter(4, [lo, hi], 'bp', fs=sr, output='sos')
        i0 = int(tt*sr); seg = ss.sosfilt(sos, y[i0:i0+int(0.05*sr)])
        return float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
    hits = []
    for tt in onsets:
        k, s, h = band(30,120,tt), band(150,800,tt), band(5000,10000,tt)
        tot = k+s+h+1e-9
        if k/tot > 0.35: hits.append((float(tt), float(tt)+0.1, 36))
        if s/tot > 0.35: hits.append((float(tt), float(tt)+0.1, 38))
        if h/tot > 0.20: hits.append((float(tt), float(tt)+0.05, 42))
    note_list_to_json(hits, f'{OUT}/{tag}.json')
    print(f'{tag}: {len(hits)} hits')

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'htdemucs'
    if which == 'htdemucs':
        d = 'separated/htdemucs/input_stereo'
        bp_stems(d, 'bp_htdemucs')
        pyin_bass(f'{d}/bass.wav', 'pyin_bass_htdemucs')
        drums_bands(f'{d}/drums.wav', 'drums_bands_htdemucs')
    elif which == 'ft':
        d = 'separated/htdemucs_ft/input_stereo'
        bp_stems(d, 'bp_ftstems')
        pyin_bass(f'{d}/bass.wav', 'pyin_bass_ft')
        drums_bands(f'{d}/drums.wav', 'drums_bands_ft')
