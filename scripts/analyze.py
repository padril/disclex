from pathlib import Path
from collections import Counter
from itertools import combinations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Delta:
    observation: str
    sample: str
    nlld: float

type AssignmentLexicon = dict[str, str]
type GroupingLexicon = set[tuple[str, str]]

def make_deltas(surfaces_path: Path | str,
                dp_updates_path: Path | str,
                ur_indices_path: Path | str) -> list[Delta]:
    surfaces = Path(surfaces_path).open('r').read()
    surfaces = surfaces.strip().split('\n')
    sr_dict = {}
    for i, s in enumerate(surfaces):
        sr_dict[i] = s.strip().replace(' ', '')

    ur_indices = Path(ur_indices_path).open('r').read()
    ur_indices = ur_indices.strip().split('\n')[1:]
    ur_indices = [f.strip().split() for f in ur_indices]
    ur_dict = {}
    for f in ur_indices:
        if len(f) == 1: f.append('')
        i, f_ = f
        ur_dict[int(i)] = f_

    dp_updates = Path(dp_updates_path).open('r').read()
    dp_updates = dp_updates.strip().split('\n')[1:]
    dp_updates = [d.strip().split() for d in dp_updates]
    dp_updates = [Delta(sr_dict[int(sr)], ur_dict[int(ur)], float(nlld))
                  for _, sr, ur, nlld in dp_updates]

    return dp_updates

def parse_assignment_lexicon(file: Path | str) -> AssignmentLexicon:
    file = Path(file)
    lines = file.open().read().strip().split('\n')
    lex = {}
    for line in lines:
        sr, ur = line.strip().split()
        lex[sr] = ur
    return lex

def write_best_lexicon(deltas: list[Delta], out_path: Path | str):
    counts: dict[str, Counter[str]] = {}
    for d in deltas:
        if d.observation not in counts:
            counts[d.observation] = Counter()
        counts[d.observation][d.sample] += 1
    
    out = Path(out_path).open('w')
    for sr, count in counts.items():
        ur, _ = count.most_common()[0]
        print(f'{sr}\t{ur}', file=out)

def initial_assignment_lexicon(surfaces_path: Path | str) -> AssignmentLexicon:
    surfaces = Path(surfaces_path).open('r').read()
    surfaces = surfaces.strip().split('\n')
    lex = {}
    for line in surfaces:
        sr = line.strip().replace(' ', '')
        lex[sr] = sr
    return lex

def grouping_lexicon(assignments: AssignmentLexicon) -> GroupingLexicon:
    groups = {}
    for sr, ur in assignments.items():
        if ur not in groups:
            groups[ur] = set()
        groups[ur] |= {sr}
    lex = set()
    for group in groups.values():
        lex |= set(combinations(group, 2))
    return lex

def scores(ref: GroupingLexicon, hyp: GroupingLexicon):
    common = {(a, b) for (a, b) in ref if (a, b) in hyp or (b, a) in hyp}

    p = len(common) / len(hyp) if hyp else 0
    r = len(common) / len(ref) if ref else 0
    f = 2 * (p * r) / (p + r) if not p == r == 0 else 0

    return p, r, f

def main(args: list[str]):
    surfaces, dp_updates, ur_indices, gold_lexicon = args

    deltas = make_deltas(surfaces, dp_updates, ur_indices)
    burnin = 0.5
    burnin_index = int(len(deltas) * burnin)
    burnin_deltas = deltas[:burnin_index]
    real_deltas = deltas[burnin_index:]
    write_best_lexicon(real_deltas, 'gibbs_lexicon.tsv')

    hyp_assignments = initial_assignment_lexicon(surfaces)
    ref_assignments = parse_assignment_lexicon(gold_lexicon)
    hyp_narrowed = {k: v for k, v in hyp_assignments.items()
                    if k in ref_assignments}
    ref_narrowed = {k: v for k, v in ref_assignments.items()
                    if k in hyp_assignments}

    ref = grouping_lexicon(ref_narrowed)

    for delta in burnin_deltas:
        if delta.observation in hyp_narrowed:
            hyp_narrowed[delta.observation] = delta.sample

    ps = []
    rs = []
    fs = []
    for delta in real_deltas:
        if delta.observation in hyp_narrowed:
            hyp_narrowed[delta.observation] = delta.sample
        hyp = grouping_lexicon(hyp_narrowed)
        p, r, f = scores(ref, hyp)
        ps.append(p)
        rs.append(r)
        fs.append(f)

    _, ax1 = plt.subplots()

    plt.plot(ps, label="Precision")
    plt.plot(rs, label="Recall")
    plt.plot(fs, label="F-sore")
    ax1.set_ylim(0, 1)

    lines = ax1.get_lines()
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(lines, labels)

    plt.savefig("prf.png", dpi=300)
    plt.close()

    farray = np.array(fs)
    print(f'mean f = {farray.mean()} with std = {farray.std()}')

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
    
