from pathlib import Path
from collections import Counter, defaultdict, deque
from itertools import combinations, tee
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Levenshtein import distance
from typing import Generator, Iterable

@dataclass
class Delta:
    step: int
    observation: str
    sample: str
    nlld: float

type AssignmentLexicon = dict[str, str]
type GroupingLexicon = set[tuple[str, str]]

def make_deltas(surfaces_path: Path | str,
                dp_updates_path: Path | str,
                ur_indices_path: Path | str) -> tuple[Generator[Delta], int]:
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

    # we intentionally don't close this so it doesn't go out of scope, and
    # that should be okay since we only leak a single file
    dp_updates = Path(dp_updates_path).open('r')
    next(dp_updates)  # remove header
    counting, yielding = tee(dp_updates)
    n = sum(1 for _ in counting)
    def gen():
        for line in yielding:
            _, step, sr_idx, ur_idx, nlld = line.strip().split()
            yield Delta(int(step),
                        sr_dict[int(sr_idx)],
                        ur_dict[int(ur_idx)],
                        float(nlld))
    return gen(), n

def parse_assignment_lexicon(file: Path | str) -> AssignmentLexicon:
    file = Path(file)
    lines = file.open().read().strip().split('\n')
    lex = {}
    for line in lines:
        sr, ur = line.strip().split()
        lex[sr] = ur
    return lex

def tail[T](it: Iterable[T], n: int):
    return iter(deque(it, maxlen=n))

def write_best_lexicon(deltas: Iterable[Delta], out_path: Path | str):
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

def ned(t: str, u: str):
    return distance(t, u) / max(len(t), len(u))

def scores(ref: AssignmentLexicon, hyp: AssignmentLexicon):
    # Precision as WNES
    clusters = defaultdict(list)
    for k_i, i in hyp.items(): clusters[i].append(k_i)
    wnes_num = 0
    wneq_num = 0
    den = 0
    for k in clusters.values():
        if len(k) <= 1:
            continue
        ned_i = 0
        neq_i = 0
        for t, u in combinations(k, 2):
            ned_i += ned(t, u)
            neq_i += ref[t] != ref[u]
        n = len(k)
        pairs = n * (n - 1) // 2
        wnes_num += (n / pairs) * ned_i
        wneq_num += (n / pairs) * neq_i
        den += n
    wnes = 1 - wnes_num / den if den != 0 else 0
    wneq = 1 - wneq_num / den if den != 0 else 0

    # Recall as iWNES
    classes = defaultdict(list)
    for c_i, i in ref.items(): classes[i].append(hyp[c_i])
    iwnes_num = 0
    iwneq_num = 0
    iden = 0
    for c in classes.values():
        if len(c) <= 1:
            continue
        ined_i = 0
        ineq_i = 0
        for t, u in combinations(c, 2):
            ined_i += ned(t, u)
            ineq_i += t != u
        n = len(c)
        pairs = n * (n - 1) // 2
        iwnes_num += (n / pairs) * ined_i
        iwneq_num += (n / pairs) * ineq_i
        iden += n
    iwnes = 1 - iwnes_num / iden if iden != 0 else 0
    iwneq = 1 - iwneq_num / iden if iden != 0 else 0

    nes_f = 2 * (wnes * iwnes) / (wnes + iwnes) if not wnes == iwnes == 0 else 0
    neq_f = 2 * (wneq * iwneq) / (wneq + iwneq) if not wneq == iwneq == 0 else 0

    return wnes, iwnes, nes_f, wneq, iwneq, neq_f

def scores_unweighted(ref: GroupingLexicon, hyp: GroupingLexicon):
    common = {(a, b) for (a, b) in ref if (a, b) in hyp or (b, a) in hyp}

    p = len(common) / len(hyp) if hyp else 0
    r = len(common) / len(ref) if ref else 0
    f = 2 * (p * r) / (p + r) if not p == r == 0 else 0

    return p, r, f

def main(args: list[str]):
    surfaces, dp_updates, ur_indices, gold_lexicon = args

    deltas, n = make_deltas(surfaces, dp_updates, ur_indices)
    burnin_ratio = 0.5
    burnin = int(n * burnin_ratio)
    writing, scoring = tee(deltas)
    write_best_lexicon(tail(writing, burnin), 'gibbs_lexicon.tsv')

    hyp = initial_assignment_lexicon(surfaces)
    ref = parse_assignment_lexicon(gold_lexicon)
    hyp = {k: v for k, v in hyp.items() if k in ref}
    ref = {k: v for k, v in ref.items() if k in hyp}

    # TODO(padril): renaming and reformatting for this whole file
    ps, rs, fs = [], [], []
    bps, brs, bfs = [], [], []
    ups, urs, ufs = [], [], []

    current_step = 0

    grouping_lexicon_ref = grouping_lexicon(ref)

    for delta in tqdm(scoring, total=n):
        if delta.observation in hyp:
            hyp[delta.observation] = delta.sample

        if delta.step > current_step:
            p, r, f, bp, br, bf = scores(ref, hyp)
            up, ur, uf = scores_unweighted(grouping_lexicon_ref, grouping_lexicon(hyp))
            ps.append(p)
            rs.append(r)
            fs.append(f)
            bps.append(bp)
            brs.append(br)
            bfs.append(bf)
            ups.append(up)
            urs.append(ur)
            ufs.append(uf)
            current_step = delta.step

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 4))

    ax1.plot(ps, label="Precision (WNES)")
    ax1.plot(rs, label="Recall (iWNES)")
    ax1.plot(fs, label="F-sore")
    ax1.set_ylim(-0.05, 1.05)

    lines = ax1.get_lines()
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(lines, labels)

    ax2.plot(bps, label="Precision (W≠)")
    ax2.plot(brs, label="Recall (iW≠)")
    ax2.plot(bfs, label="F-sore")
    ax2.set_ylim(-0.05, 1.05)

    lines = ax2.get_lines()
    labels = [str(line.get_label()) for line in lines]
    ax2.legend(lines, labels)

    ax3.plot(ups, label="Precision (grouping)")
    ax3.plot(urs, label="Recall (grouping)")
    ax3.plot(ufs, label="F-sore")
    ax3.set_ylim(-0.05, 1.05)

    lines = ax3.get_lines()
    labels = [str(line.get_label()) for line in lines]
    ax3.legend(lines, labels)

    fig.savefig("prf.svg")

    parray = np.fromiter(tail(ps, burnin), dtype=float)
    rarray = np.fromiter(tail(rs, burnin), dtype=float)
    farray = np.fromiter(tail(fs, burnin), dtype=float)
    print(f'μ NES F = {farray.mean()} with σ = {farray.std()}')
    print(f'\tμ WNES = {parray.mean()} with σ = {parray.std()}')
    print(f'\tμ iWNES = {rarray.mean()} with σ = {rarray.std()}')
    bparray = np.fromiter(tail(bps, burnin), dtype=float)
    brarray = np.fromiter(tail(brs, burnin), dtype=float)
    bfarray = np.fromiter(tail(bfs, burnin), dtype=float)
    print(f'μ ≠ F = {bfarray.mean()} with σ = {bfarray.std()}')
    print(f'\tμ W≠ = {bparray.mean()} with σ = {bparray.std()}')
    print(f'\tμ iW≠ = {brarray.mean()} with σ = {brarray.std()}')
    uparray = np.fromiter(tail(ups, burnin), dtype=float)
    urarray = np.fromiter(tail(urs, burnin), dtype=float)
    ufarray = np.fromiter(tail(ufs, burnin), dtype=float)
    print(f'μ Grouping F = {ufarray.mean()} with σ = {ufarray.std()}')
    print(f'\tμ Grouping P = {uparray.mean()} with σ = {uparray.std()}')
    print(f'\tμ Grouping R = {urarray.mean()} with σ = {urarray.std()}')

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
    
