import numpy as np
from midi_utils import FeatureBundle

# ── helper similarities (Section 4.3 equations) ───────────────────
def _motif_interval_similarity(g: FeatureBundle, s: FeatureBundle):
    gi = np.diff([n.pitch.midi for n in g.melody])
    si = np.diff([n.pitch.midi for n in s.melody])
    if not gi.size or not si.size:
        return 0.0
    m = min(len(gi), len(si))
    return max(0, 1 - np.linalg.norm(gi[:m] - si[:m]) / (12 * m))

def _harmonic_similarity(g: FeatureBundle, s: FeatureBundle):
    groots = {c.root().pitchClass for c in g.chords}
    sroots = {c.root().pitchClass for c in s.chords}
    return len(groots & sroots) / max(1, len(groots))

def _rhythmic_similarity(g: FeatureBundle, s: FeatureBundle):
    if not g.rhythm or not s.rhythm:
        return 0.0
    m = min(len(g.rhythm), len(s.rhythm))
    var = np.var(np.subtract(g.rhythm[:m], s.rhythm[:m]))
    base = np.mean(s.rhythm)**2 + 1e-6
    return max(0, 1 - var / base)

def _chromatic_score(g, s1, s2, w1, w2):
    target = w1 * np.mean(s1.chromatic_intervals or [0]) + w2 * np.mean(s2.chromatic_intervals or [0])
    gen = np.mean(g.chromatic_intervals or [0])
    return 1.0 if target == 0 else max(0, 1 - abs(gen - target) / target)

# ── top‑level fitness ──────────────────────────────────────────────
def fitness(gen: FeatureBundle, s1: FeatureBundle, s2: FeatureBundle, w1, w2):
    M = w1 * _motif_interval_similarity(gen, s1) + w2 * _motif_interval_similarity(gen, s2)
    H = w1 * _harmonic_similarity(gen, s1) + w2 * _harmonic_similarity(gen, s2)
    R = w1 * _rhythmic_similarity(gen, s1) + w2 * _rhythmic_similarity(gen, s2)
    C = _chromatic_score(gen, s1, s2, w1, w2)
    return 0.30*M + 0.30*H + 0.30*R + 0.10*C
