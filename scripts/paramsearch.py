import subprocess
from random import sample
from pathlib import Path
import time
import numpy as np
from analyze import (
        make_deltas,
        initial_assignment_lexicon,
        parse_assignment_lexicon,
        grouping_lexicon,
        scores
        )

observations = 'data/valid_observations.txt'
alignments = 'data/valid_alignments.txt'
gold_lexicon = 'data/valid_lexicon.tsv'

phonemes = 'data/phones.txt'

dp_updates = 'out/dp_updates.out'
ur_indices = 'out/ur_indices.out'

steps = 50
rebuild_stride = 5

def run_for_parameters(alpha, ug_mix, n_ratio):
    observation_list = Path(observations).open('r').read().strip().split('\n')
    alignments_list = Path(alignments).open('r').read().strip().split('\n')
    n = len(observation_list)
    narrowed_n = int(n * n_ratio)
    narrowed_list = sample(observation_list, narrowed_n)
    narrowed = '/tmp/narrowed.txt'
    with Path(narrowed).open('w') as f:
        for line in narrowed_list:
            print(line, file=f)
 
    seed = int((time.time() * 1e7) % 1e7)
    
    command = ['./build/bin/disclex',
               '--prior', 'fst',
               '--phones', phonemes,
               '--alignments', narrowed,
               '--phonemes', phonemes,
               '--steps', str(steps),
               '--rebuild-stride', str(rebuild_stride),
               '--alpha', str(alpha),
               '--universal-grammar-weight', str(ug_mix),
               '--output-deltas', dp_updates,
               '--output-parameters', ur_indices,
               '--output-fsts-dir', 'out/fsts/'
               '--seed', str(seed)]
    print(' '.join(command))
    subprocess.run(command)

    deltas = make_deltas(narrowed, dp_updates, ur_indices)
    burnin = 0.5
    burnin_index = int(len(deltas) * burnin)
    burnin_deltas = deltas[:burnin_index]
    real_deltas = deltas[burnin_index:]

    hyp_assignments = initial_assignment_lexicon(narrowed)
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

def main():
    out = open('out/paramsearch.out', 'w')
    observation_list = Path(observations).open('r').read().strip().split('\n')
    true_n = len(observation_list)
    for alpha in [4.0, 5.0, 6.0, 7.0, 8.0]:
        for ug_mix in [0.05, 0.25, 0.45, 0.65, 0.85, 1.05]:
            for n_ratio in [1.0]:
                for chain in [1, 2, 3]:
                    mean_f, std_f = run_for_parameters(alpha, ug_mix, n_ratio)
                    n = int(true_n * n_ratio)
                    print(f'chain={chain}, '
                          f'alpha={alpha:0.2f}, '
                          f'ug_mix={ug_mix:0.2f}, '
                          f'n={n}, '
                          f'mean_f={mean_f:0.2f}, '
                          f'std_f={std_f:0.2f}',
                          file=out,
                          flush=True)

if __name__ == '__main__':
    main()
