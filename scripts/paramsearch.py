import subprocess
import numpy as np
from analyze import *

observations = 'data/valid_observations.txt'
gold_lexicon = 'data/valid_lexicon.tsv'

phonemes = 'data/phones.txt'

dp_updates = 'out/dp_updates.out'
ur_indices = 'out/ur_indices.out'

steps = 100
rebuild_stride = 10

def run_for_parameters(alpha, ug_mix):
    subprocess.run(['./build/bin/disclex',
                    '--prior', 'fst',
                    '--phones', phonemes,
                    '--observations', observations, 
                    '--phonemes', phonemes,
                    '--steps', str(steps),
                    '--rebuild-stride', str(rebuild_stride),
                    '--alpha', str(alpha),
                    '--universal-grammar-weight', str(ug_mix),
                    '--output-deltas', dp_updates,
                    '--output-parameters', ur_indices])

    deltas = make_deltas(observations, dp_updates, ur_indices)
    burnin = 0.5
    burnin_index = int(len(deltas) * burnin)
    burnin_deltas = deltas[:burnin_index]
    real_deltas = deltas[burnin_index:]

    hyp_assignments = initial_assignment_lexicon(observations)
    ref_assignments = parse_assignment_lexicon(gold_lexicon)
    hyp_narrowed = {k: v for k, v in hyp_assignments.items()
                    if k in ref_assignments}
    ref_narrowed = {k: v for k, v in ref_assignments.items()
                    if k in hyp_assignments}

    ref = grouping_lexicon(ref_narrowed)

    for delta in burnin_deltas:
        if delta.observation in hyp_narrowed:
            hyp_narrowed[delta.observation] = delta.sample

    fs = []
    for delta in real_deltas:
        if delta.observation in hyp_narrowed:
            hyp_narrowed[delta.observation] = delta.sample
        hyp = grouping_lexicon(hyp_narrowed)
        _, _, f = scores(ref, hyp)
        fs.append(f)

    farray = np.array(fs)
    return farray.mean(), farray.std()

if __name__ == '__main__':
    out = open('out/paramsearch.out', 'w')
    for alpha in [3.0, 5.0, 10.0, 25.0, 50.0]:
        for ug_mix in [1.0, 3.0, 5.0, 7.0]:
            mean_f, std_f = run_for_parameters(alpha, ug_mix)
            print(f'alpha={alpha:0.2f}, '
                  f'ug_mix={ug_mix:0.2f}, '
                  f'mean_f={mean_f:0.2f}, '
                  f'std_f={std_f:0.2f}',
                  file=out)


