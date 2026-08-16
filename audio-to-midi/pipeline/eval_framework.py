"""Evaluation framework for audio-to-MIDI transcription candidates.

Provides:
- MIDI note loading (drums excluded by default) suitable for mir_eval.
- Pairwise note-level agreement between candidate MIDI transcriptions.
- Audio-vs-render metrics: bidirectional onset P/R/F, per-band envelope
  correlation, and tempo-invariant rhythm autocorrelation similarity.
- A compact text report over a set of candidates and renders.

All functions are pure (no global state).
"""

import numpy as np
import librosa
import scipy.signal
import pretty_midi
import mir_eval

MIN_DUR = 0.01  # mir_eval.transcription requires offset > onset
_SR = 22050


# ---------------------------------------------------------------------------
# 1. MIDI loading
# ---------------------------------------------------------------------------

def load_notes(midi_path, include_drums=False):
    """Load note events from a MIDI file.

    Returns (intervals, pitches):
      intervals: float ndarray (N, 2) of [onset, offset] seconds, sorted by
                 onset, with minimum duration clamped to MIN_DUR.
      pitches:   float ndarray (N,) of pitch values in Hz.
    Drum instruments are excluded unless include_drums=True.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    onsets, offsets, midis = [], [], []
    for inst in pm.instruments:
        if inst.is_drum and not include_drums:
            continue
        for note in inst.notes:
            onsets.append(note.start)
            offsets.append(max(note.end, note.start + MIN_DUR))
            midis.append(note.pitch)
    if not onsets:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)
    intervals = np.column_stack([onsets, offsets]).astype(float)
    pitches = librosa.midi_to_hz(np.asarray(midis, dtype=float))
    order = np.argsort(intervals[:, 0], kind="stable")
    return intervals[order], np.asarray(pitches, dtype=float)[order]


def load_drum_onsets(midi_path):
    """Return sorted ndarray of onset times (s) from drum instruments only."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    onsets = [note.start
              for inst in pm.instruments if inst.is_drum
              for note in inst.notes]
    return np.sort(np.asarray(onsets, dtype=float))


# ---------------------------------------------------------------------------
# 2/3. Note-level agreement
# ---------------------------------------------------------------------------

def note_agreement(midi_a, midi_b, onset_tol=0.05):
    """Note-level agreement between two MIDI files (a = reference, b = estimate).

    Returns dict with precision, recall, f1 (onset+pitch+offset, default
    offset_ratio) and precision_no_offset, recall_no_offset, f1_no_offset
    (onset+pitch only). f1_no_offset is the primary number.
    """
    ref_i, ref_p = load_notes(midi_a)
    est_i, est_p = load_notes(midi_b)
    zeros = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
             "precision_no_offset": 0.0, "recall_no_offset": 0.0,
             "f1_no_offset": 0.0}
    if len(ref_p) == 0 or len(est_p) == 0:
        return zeros
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, onset_tolerance=onset_tol)
    pn, rn, fn, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, onset_tolerance=onset_tol,
        offset_ratio=None)
    return {"precision": float(p), "recall": float(r), "f1": float(f),
            "precision_no_offset": float(pn), "recall_no_offset": float(rn),
            "f1_no_offset": float(fn)}


def agreement_matrix(midi_paths):
    """Dict-of-dicts {a: {b: f1_no_offset}} over all ordered pairs.

    Diagonal is included (self-agreement, normally 1.0).
    """
    matrix = {}
    for a in midi_paths:
        matrix[a] = {}
        for b in midi_paths:
            matrix[a][b] = note_agreement(a, b)["f1_no_offset"]
    return matrix


# ---------------------------------------------------------------------------
# 4. Bidirectional onset P/R/F on audio
# ---------------------------------------------------------------------------

def _load_audio(path):
    y, sr = librosa.load(path, sr=_SR, mono=True)
    return y, sr


def _detect_onsets(path):
    y, sr = _load_audio(path)
    return librosa.onset.onset_detect(y=y, sr=sr, backtrack=False,
                                      units="time")


def onset_prf_bidirectional(reference_wav, estimate_wav, tol=0.05):
    """Onset precision/recall/F1: reference audio onsets vs render onsets."""
    ref_onsets = np.asarray(_detect_onsets(reference_wav), dtype=float)
    est_onsets = np.asarray(_detect_onsets(estimate_wav), dtype=float)
    if len(ref_onsets) == 0 or len(est_onsets) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    f, p, r = mir_eval.onset.f_measure(ref_onsets, est_onsets, window=tol)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


# ---------------------------------------------------------------------------
# 5. Band-limited envelope correlation
# ---------------------------------------------------------------------------

def _band_rms(y, sr, band):
    """RMS envelope of y filtered into a band: 'low', 'mid', or 'high'."""
    if band == "low":
        sos = scipy.signal.butter(4, 200, btype="lowpass", fs=sr, output="sos")
    elif band == "mid":
        sos = scipy.signal.butter(4, [200, 2000], btype="bandpass", fs=sr,
                                  output="sos")
    elif band == "high":
        sos = scipy.signal.butter(4, 4000, btype="highpass", fs=sr,
                                  output="sos")
    else:
        raise ValueError(f"unknown band: {band}")
    filtered = scipy.signal.sosfilt(sos, y)
    return librosa.feature.rms(y=filtered)[0]


def band_envelope_corr(reference_wav, estimate_wav):
    """Pearson correlation of RMS envelopes in low/mid/high bands."""
    y_ref, sr = _load_audio(reference_wav)
    y_est, _ = _load_audio(estimate_wav)
    out = {}
    for band in ("low", "mid", "high"):
        e_ref = _band_rms(y_ref, sr, band)
        e_est = _band_rms(y_est, sr, band)
        n = min(len(e_ref), len(e_est))
        if n < 2:
            out[band] = 0.0
            continue
        a, b = e_ref[:n], e_est[:n]
        if np.std(a) == 0 or np.std(b) == 0:
            out[band] = 0.0
        else:
            out[band] = float(np.corrcoef(a, b)[0, 1])
    return out


# ---------------------------------------------------------------------------
# 6. Rhythm autocorrelation similarity
# ---------------------------------------------------------------------------

def _norm_autocorr(env, max_lag):
    """Positive-lag autocorrelation of env up to max_lag frames, normalized
    by its lag-0 value."""
    env = np.asarray(env, dtype=float)
    if len(env) == 0:
        return np.zeros(max_lag, dtype=float)
    full = np.correlate(env, env, mode="full")
    center = len(env) - 1
    ac = full[center:center + max_lag]
    if len(ac) < max_lag:
        ac = np.pad(ac, (0, max_lag - len(ac)))
    if ac[0] > 0:
        ac = ac / ac[0]
    return ac


def rhythm_autocorr_sim(reference_wav, estimate_wav):
    """Cosine similarity in [0, 1] of normalized onset-strength
    autocorrelations (positive lags up to 4 s). Tempo-invariant-ish rhythm
    texture similarity."""
    y_ref, sr = _load_audio(reference_wav)
    y_est, _ = _load_audio(estimate_wav)
    hop = 512
    max_lag = int(round(4.0 * sr / hop))
    env_ref = librosa.onset.onset_strength(y=y_ref, sr=sr, hop_length=hop)
    env_est = librosa.onset.onset_strength(y=y_est, sr=sr, hop_length=hop)
    ac_ref = _norm_autocorr(env_ref, max_lag)
    ac_est = _norm_autocorr(env_est, max_lag)
    na, nb = np.linalg.norm(ac_ref), np.linalg.norm(ac_est)
    if na == 0 or nb == 0:
        return 0.0
    sim = float(np.dot(ac_ref, ac_est) / (na * nb))
    return float(np.clip(sim, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 7. Combined render evaluation
# ---------------------------------------------------------------------------

def evaluate_render(reference_wav, estimate_wav):
    """Merge onset P/R/F, band envelope correlations, and rhythm similarity."""
    out = dict(onset_prf_bidirectional(reference_wav, estimate_wav))
    env = band_envelope_corr(reference_wav, estimate_wav)
    out.update({f"env_{k}": v for k, v in env.items()})
    out["rhythm_sim"] = rhythm_autocorr_sim(reference_wav, estimate_wav)
    return out


# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------

def report(candidates, renders):
    """Print agreement matrix over candidate MIDIs and render metrics.

    candidates: {label: midi_path}
    renders:    {label: (reference_wav, rendered_wav)}
    """
    labels = list(candidates.keys())
    if labels:
        paths = [candidates[k] for k in labels]
        mat = agreement_matrix(paths)
        w = max(12, max(len(s) for s in labels) + 2)
        print("Note agreement (F1, onset+pitch, 50 ms tolerance)")
        print("row=reference, col=estimate")
        header = " " * w + "".join(f"{lab:>{w}}" for lab in labels)
        print(header)
        for la in labels:
            row = f"{la:<{w}}"
            for lb in labels:
                row += f"{mat[candidates[la]][candidates[lb]]:>{w}.3f}"
            print(row)
        print()

    if renders:
        cols = ["precision", "recall", "f1", "env_low", "env_mid",
                "env_high", "rhythm_sim"]
        lw = max(12, max(len(s) for s in renders) + 2)
        print("Render vs reference audio")
        print(f"{'':{lw}}" + "".join(f"{c:>11}" for c in cols))
        for label, (ref_wav, est_wav) in renders.items():
            res = evaluate_render(ref_wav, est_wav)
            row = f"{label:<{lw}}"
            for c in cols:
                row += f"{res[c]:>11.3f}"
            print(row)
        print()


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    work = os.path.dirname(os.path.abspath(__file__))
    candidates = {
        "v42n": os.path.join(work, "recreation_v42n.mid"),
        "v3": os.path.join(work, "recreation_v3.mid"),
        "v1_messy": os.path.join(work, "v1_messy.mid"),
        "basicpitch": os.path.join(work, "fullmix_basicpitch.mid"),
    }
    ref = os.path.join(work, "input_mono.wav")
    renders = {
        "v3": (ref, os.path.join(work, "rendered_v3.wav")),
        "v42n": (ref, os.path.join(work, "rendered_v42n.wav")),
    }
    report(candidates, renders)
