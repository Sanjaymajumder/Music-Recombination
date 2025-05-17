import random
from typing import List
from midi_utils import FeatureBundle
from genetic_ops import build_child
from fitness import fitness

def init_population(src1: FeatureBundle, src2: FeatureBundle,
                    size=200) -> List[FeatureBundle]:
    return [build_child(src1, src2) for _ in range(size)]

def evolve(pop: List[FeatureBundle],
           src1: FeatureBundle,
           src2: FeatureBundle,
           w1, w2,
           generations=25,
           elite_ratio=0.2):
    for _ in range(generations):
        pop.sort(key=lambda ind: fitness(ind, src1, src2, w1, w2), reverse=True)
        elite = pop[:max(2, int(len(pop)*elite_ratio))]
        while len(elite) < len(pop):
            p1, p2 = random.sample(elite, 2)
            elite.append(build_child(p1, p2))
        pop = elite
    return max(pop, key=lambda ind: fitness(ind, src1, src2, w1, w2))
