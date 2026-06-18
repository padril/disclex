from pathlib import Path
import math
from collections import Counter, defaultdict
from itertools import combinations
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Levenshtein import distance

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
    dp_updates = [Delta(int(step), sr_dict[int(sr)], ur_dict[int(ur)], float(nlld))
                  for _, step, sr, ur, nlld in dp_updates]

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

def ned(t: str, u: str):
    return distance(t, u) / max(len(t), len(u))

def scores(ref: AssignmentLexicon, hyp: AssignmentLexicon, cluster_ned, class_ned):
    # Precision as WNES
    clusters = defaultdict(list)
    for k_i, i in hyp.items(): clusters[i].append(k_i)
    wnes_num = 0
    wnes_den = 0
    for k in clusters.values():
        if len(k) <= 1:
            continue
        ned_i = sum(cluster_ned(t, u) for t, u in combinations(k, 2))
        wnes_num += (len(k) / math.comb(len(k), 2)) * ned_i
        wnes_den += len(k)
    p = 1 - wnes_num / wnes_den

    # Recall as iWNES
    classes = defaultdict(list)
    for c_i, i in ref.items(): classes[i].append(hyp[c_i])
    iwnes_num = 0
    iwnes_den = 0
    for c in classes.values():
        if len(c) <= 1:
            continue
        ned_i = sum(class_ned(t, u) for t, u in combinations(c, 2))
        iwnes_num += (len(c) / math.comb(len(c), 2)) * ned_i
        iwnes_den += len(c)
    r = 1 - iwnes_num / iwnes_den

    f = 2 * (p * r) / (p + r) if not p == r == 0 else 0

    return p, r, f

def scores_unweighted(ref: GroupingLexicon, hyp: GroupingLexicon):
    common = {(a, b) for (a, b) in ref if (a, b) in hyp or (b, a) in hyp}

    p = len(common) / len(hyp) if hyp else 0
    r = len(common) / len(ref) if ref else 0
    f = 2 * (p * r) / (p + r) if not p == r == 0 else 0

    return p, r, f

def main(args: list[str]):
    surfaces, dp_updates, ur_indices, gold_lexicon = args

    deltas = make_deltas(surfaces, dp_updates, ur_indices)
    burnin = 0.5
    write_best_lexicon(deltas[int(len(deltas) * burnin):], 'gibbs_lexicon.tsv')

    hyp_assignments = initial_assignment_lexicon(surfaces)
    ref_assignments = parse_assignment_lexicon(gold_lexicon)
    hyp = {k: v for k, v in hyp_assignments.items()
                    if k in ref_assignments}
    ref = {k: v for k, v in ref_assignments.items()
                    if k in hyp_assignments}

    ps = []
    rs = []
    fs = []
    temp_ps = []
    temp_rs = []
    temp_fs = []
    bin_ps = []
    bin_rs = []
    bin_fs = []
    temp_bin_ps = []
    temp_bin_rs = []
    temp_bin_fs = []
    ups = []
    urs = []
    ufs = []
    temp_ups = []
    temp_urs = []
    temp_ufs = []

    current_step = 0

    for delta in tqdm(deltas):
        if delta.observation in hyp:
            hyp[delta.observation] = delta.sample
        p, r, f = scores(ref, hyp, ned, ned)
        temp_ps.append(p)
        temp_rs.append(r)
        temp_fs.append(f)
        bp, br, bf = scores(ref, hyp, lambda t, u: ref[t] != ref[u], lambda t, u: t != u)
        temp_bin_ps.append(bp)
        temp_bin_rs.append(br)
        temp_bin_fs.append(bf)
        up, ur, uf = scores_unweighted(grouping_lexicon(ref), grouping_lexicon(hyp))
        temp_ups.append(up)
        temp_urs.append(ur)
        temp_ufs.append(uf)

        if delta.step > current_step:
            ps.append(sum(temp_ps) / len(temp_ps))
            rs.append(sum(temp_rs) / len(temp_rs))
            fs.append(sum(temp_fs) / len(temp_fs))
            bin_ps.append(sum(temp_bin_ps) / len(temp_bin_ps))
            bin_rs.append(sum(temp_bin_rs) / len(temp_bin_rs))
            bin_fs.append(sum(temp_bin_fs) / len(temp_bin_fs))
            ups.append(sum(temp_ups) / len(temp_ups))
            urs.append(sum(temp_urs) / len(temp_urs))
            ufs.append(sum(temp_ufs) / len(temp_ufs))
            temp_ps = []
            temp_rs = []
            temp_fs = []
            temp_bin_ps = []
            temp_bin_rs = []
            temp_bin_fs = []
            temp_ups = []
            temp_urs = []
            temp_ufs = []
            current_step = delta.step

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 4))

    ax1.plot(ps, label="Precision (WNES)")
    ax1.plot(rs, label="Recall (iWNES)")
    ax1.plot(fs, label="F-sore")
    ax1.set_ylim(-0.05, 1.05)

    lines = ax1.get_lines()
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(lines, labels)

    ax2.plot(bin_ps, label="Precision (W≠)")
    ax2.plot(bin_rs, label="Recall (iW≠)")
    ax2.plot(bin_fs, label="F-sore")
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

    parray = np.array(ps[int(len(ps) * burnin):])
    rarray = np.array(rs[int(len(rs) * burnin):])
    farray = np.array(fs[int(len(fs) * burnin):])
    print(f'μ NES F = {farray.mean()} with σ = {farray.std()}')
    print(f'\tμ WNES = {parray.mean()} with σ = {parray.std()}')
    print(f'\tμ iWNES = {rarray.mean()} with σ = {rarray.std()}')
    bparray = np.array(bin_ps[int(len(bin_ps) * burnin):])
    brarray = np.array(bin_rs[int(len(bin_rs) * burnin):])
    bfarray = np.array(bin_fs[int(len(bin_fs) * burnin):])
    print(f'μ ≠ F = {bfarray.mean()} with σ = {bfarray.std()}')
    print(f'\tμ W≠ = {bparray.mean()} with σ = {bparray.std()}')
    print(f'\tμ iW≠ = {brarray.mean()} with σ = {brarray.std()}')
    uparray = np.array(ups[int(len(ps) * burnin):])
    urarray = np.array(urs[int(len(rs) * burnin):])
    ufarray = np.array(ufs[int(len(fs) * burnin):])
    print(f'μ Grouping F = {ufarray.mean()} with σ = {ufarray.std()}')
    print(f'\tμ Grouping P = {uparray.mean()} with σ = {uparray.std()}')
    print(f'\tμ Grouping R = {urarray.mean()} with σ = {urarray.std()}')

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
    
