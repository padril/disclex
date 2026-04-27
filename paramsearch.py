from scipy.optimize import dual_annealing
import subprocess
from analyze import *


#         alpha    ug_mix
bounds = [(0.01, 10), (0.0, 2.0)]

count = 0

def small_call(x):
    global count
    print(f'calls so far: {count}')
    count += 1

    gold_lexicon = "gold_lexicon_1h.tsv"
    ref = parse_lexicon(gold_lexicon)

    alpha, ug_mix = x
    surfaces = "dev10.txt"
    subprocess.run(["./build/bin/disclex",
                    "--prior", "fst",
                    "--phones", "phones.txt",
                    "--observations", surfaces, 
                    "--phonemes", "phones.txt",
                    "--steps", str(20),
                    "--rebuild-stride", str(10),
                    "--alpha", str(alpha),
                    "--universal-grammar-weight", str(ug_mix)], capture_output=True)

    deltas = "dp_updates.out"
    forms = "ur_indices.out"
    deltas = parse_deltas(deltas, 0.5)
    surfaces = parse_surfaces(surfaces)
    forms = parse_forms(forms)
    counts = get_counts(forms, deltas)
    hyp = make_lexicon(surfaces, counts, 'gibbs_lexicon.tsv')
    p, r, f = grouping(ref, hyp)

    return f

print(dual_annealing(small_call, bounds, maxiter=100))
