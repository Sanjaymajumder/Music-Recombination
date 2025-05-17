from music21 import stream
from midi_utils import (
    parse_midi, allowed_scale_pitches,
    weighted_tempo, dominant_time_signature, FeatureBundle
)
from genetic_algorithm import init_population, evolve
from fitness import fitness

def bundle_to_stream(b: FeatureBundle, ts, tempo_val) -> stream.Stream:
    s = stream.Stream()
    s.append(ts)
    s.append(tempo.MetronomeMark(number=tempo_val))
    for n in b.melody:
        s.append(n)
    for c in b.chords:
        s.append(c)
    return s

def main():
    m1 = input("First MIDI path: ").strip()
    m2 = input("Second MIDI path: ").strip()
    w1 = float(input("Weight for first (0‑1): "))
    w2 = 1.0 - w1

    src1 = parse_midi(m1)
    src2 = parse_midi(m2)

    # Combined tempo & time signature
    tempo_val = weighted_tempo(src1, src2, w1, w2)
    ts = dominant_time_signature(src1, src2, w1, w2)

    pop = init_population(src1, src2)
    best = evolve(pop, src1, src2, w1, w2)

    print("Best fitness:", round(fitness(best, src1, src2, w1, w2), 4))

    out = bundle_to_stream(best, ts, tempo_val)
    out.write('midi', fp='recombined_output.mid')
    print("Saved: recombined_output.mid")

if __name__ == "__main__":
    main()
