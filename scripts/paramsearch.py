import subprocess
import numpy as np
from tqdm import tqdm
from analyze import (
        make_deltas,
        parse_assignment_lexicon,
        scores_unweighted,
        grouping_lexicon,
        )

split = 'data/splits/valid'
splits = 'data/conditions/training_splits.csv'
alignments = f'{split}/alignments.csv'
identity = f'{split}/ident.csv'
lexicon = f'{split}/lexicon.csv'
universal = 'data/naive_universal_grammar.csv'

segments = 'data/segments.txt'

parameters = 'out/parameters.out'
deltas = 'out/deltas.out'

steps = 500
rebuild_stride = 50
mh_ratio = 0

def run_for_parameters(alpha):
    command = ['./build/bin/disclex',
               '--prior', 'fst',
               '--alignments', alignments,
               '--alignments', identity,
               '--alignments', universal,
               '--splits', splits,
               '--phones', segments,
               '--phonemes', segments,
               '--steps', str(steps),
               '--rebuild-stride', str(rebuild_stride),
               '--alpha', str(alpha),
               '--mh-ratio', str(mh_ratio),
               '--output-deltas', deltas,
               '--output-parameters', parameters,
               '--output-models-dir', 'out/models/',
               ]

    subprocess.run(command)

    scoring, n = make_deltas(alignments, deltas, parameters)
    hyp = parse_assignment_lexicon(alignments)
    ref = parse_assignment_lexicon(lexicon)
    hyp = {k: v for k, v in hyp.items() if k in ref}
    ref = {k: v for k, v in ref.items() if k in hyp}

    fs = []
    current_step = 0

    for delta in tqdm(scoring, total=n):
        if delta.observation in hyp:
            hyp[delta.observation] = delta.sample

        if delta.step > current_step:
            _, _, f = scores_unweighted(grouping_lexicon(ref),
                                        grouping_lexicon(hyp))
            fs.append(f)
            current_step = delta.step

    burnin_ratio = 0.5
    burnin = int(len(fs) * burnin_ratio)
    farray = np.array(fs[burnin:])
    return farray.mean(), farray.std()

def main():
    out = open('out/paramsearch.out', 'w')
    for alpha in np.linspace(1, 50, 20, endpoint=True):
        mean_f, std_f = run_for_parameters(alpha)
        print(f'alpha={alpha:0.2f}, '
              f'µNESf={mean_f:0.2f}, '
              f'σNESf={std_f:0.2f}',
              file=out,
              flush=True)

if __name__ == '__main__':
    main()
