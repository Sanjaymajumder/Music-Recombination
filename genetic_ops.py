import random
from copy import deepcopy
from typing import List
from music21 import note, chord
from midi_utils import FeatureBundle, allowed_scale_pitches

# ── Crossover helpers ─────────────────────────────────────────────
def melody_crossover(p1: FeatureBundle, p2: FeatureBundle) -> List[note.Note]:
    cut1 = random.choice(p1.motifs)[0] if p1.motifs else len(p1.melody)//2
    cut2 = random.choice(p2.motifs)[0] if p2.motifs else len(p2.melody)//2
    return p1.melody[:cut1] + p2.melody[cut2:]

def chord_block_crossover(p1: FeatureBundle, p2: FeatureBundle) -> List[chord.Chord]:
    blk1 = random.choice(p1.chord_blocks) if p1.chord_blocks else p1.chords[:4]
    blk2 = random.choice(p2.chord_blocks) if p2.chord_blocks else p2.chords[-4:]
    return blk1 + blk2

# ── Mutation helpers ──────────────────────────────────────────────
def mutate_rhythm(rhythm, profile, rate=0.1):
    for i in range(len(rhythm)):
        if random.random() < rate:
            rhythm[i] = random.choice(profile)

def mutate_pitch(melody, scale_pitches, rate=0.05):
    for n in melody:
        if isinstance(n, note.Note) and random.random() < rate:
            n.pitch.midi = random.choice(scale_pitches)

# ── Child builder ─────────────────────────────────────────────────
def build_child(p1: FeatureBundle, p2: FeatureBundle) -> FeatureBundle:
    scale_pitches = allowed_scale_pitches(p1.key_signature)
    child = deepcopy(p1)          # copy base fields
    child.melody = melody_crossover(p1, p2)
    child.chords = chord_block_crossover(p1, p2)
    child.rhythm = [n.quarterLength for n in child.melody]
    mutate_rhythm(child.rhythm, p1.rhythm_profile)
    mutate_pitch(child.melody, scale_pitches)
    child.chromatic_intervals = [
        abs(a.pitch.midi - b.pitch.midi) for a,b in zip(child.melody, child.melody[1:])
    ]
    return child
