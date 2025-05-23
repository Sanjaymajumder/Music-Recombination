from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple
import itertools
from music21 import (
    converter, note, chord, stream, key, scale, meter, tempo
)

@dataclass
class FeatureBundle:
    melody: List[note.Note]
    motifs: List[Tuple[int, int]]
    chords: List[chord.Chord]
    chord_blocks: List[List[chord.Chord]]
    rhythm: List[float]
    rhythm_profile: List[float]
    chromatic_intervals: List[int]
    key_signature: key.Key
    time_signature: meter.TimeSignature
    tempo: float

# ------------------------------------------------------------------
def parse_midi(path: str, motif_win: int = 4) -> FeatureBundle:
    s = converter.parse(path)

    # ─ melody / chords / rhythm
    melody, chords, rhythm = [], [], []
    for el in s.recurse().notesAndRests:
        if isinstance(el, note.Note):
            melody.append(el)
            rhythm.append(el.quarterLength)
        elif isinstance(el, chord.Chord):
            chords.append(el)
            rhythm.append(el.quarterLength)

    # ─ motif detection (contour)
    motifs = []
    if len(melody) >= motif_win:
        contours = []
        for i in range(len(melody) - motif_win + 1):
            sign = tuple(
                (melody[i + j + 1].pitch.midi - melody[i + j].pitch.midi > 0) -
                (melody[i + j + 1].pitch.midi - melody[i + j].pitch.midi < 0)
                for j in range(motif_win - 1)
            )
            contours.append(sign)
        repeating = {c for c, n in Counter(contours).items() if n > 1}
        motifs = [
            (i, i + motif_win)
            for i, c in enumerate(contours) if c in repeating
        ]

    # ─ chord blocks (bars)
    chord_blocks: List[List[chord.Chord]] = []
    for prt in s.parts:
        for m in prt.getElementsByClass(stream.Measure):
            blk = [c for c in m.notes if isinstance(c, chord.Chord)]
            if blk:
                chord_blocks.append(blk)

    # ─ chromatic interval list
    chromatic = [
        abs(a.pitch.midi - b.pitch.midi)
        for a, b in zip(melody, melody[1:]) if isinstance(a, note.Note) and isinstance(b, note.Note)
    ]

    # ─ global key & time signature
    try:
        k_sig = s.analyze('key')
    except Exception:
        k_sig = key.Key('C')
    try:
        t_sig = s.recurse().getElementsByClass(meter.TimeSignature)[0]
    except Exception:
        t_sig = meter.TimeSignature('4/4')

    # ─ first tempo mark
    try:
        tempo_val = s.metronomeMarkBoundaries()[0][2].number
    except Exception:
        tempo_val = 120.0

    rhythm_profile = list(Counter(rhythm).elements())

    return FeatureBundle(
        melody, motifs, chords, chord_blocks,
        rhythm, rhythm_profile, chromatic,
        k_sig, t_sig, tempo_val
    )

# ------------------------------------------------------------------
def allowed_scale_pitches(k: key.Key):
    sc = scale.MajorScale(k.tonic) if k.mode == 'major' else scale.MinorScale(k.tonic)
    return [p.midi for p in itertools.islice(sc.getPitches(k.tonic, k.tonic.transpose('P8')), 7)]

# ------------------------------------------------------------------
def weighted_tempo(f1: FeatureBundle, f2: FeatureBundle, w1: float, w2: float) -> float:
    return (f1.tempo * w1 + f2.tempo * w2) / (w1 + w2)

def dominant_time_signature(f1: FeatureBundle, f2: FeatureBundle, w1: float, w2: float):
    if w1 > w2:
        return f1.time_signature
    if w2 > w1:
        return f2.time_signature
    # equal weights → pick most common denominator
    return f1.time_signature if f1.time_signature.ratioString >= f2.time_signature.ratioString else f2.time_signature
