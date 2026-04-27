from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from itertools import combinations

type Type = int
type Word = str
type Lexicon = dict[Word, Type]

@dataclass
class Delta:
    step: int
    index: int
    sample: int
    nlld: float

def parse_surfaces(file: Path | str) -> dict[int, str]:
    surfaces = Path(file).open('r').read()
    surfaces = surfaces.strip().split('\n')
    surfaces_ = surfaces
    surfaces = {}
    for i, s in enumerate(surfaces_):
        surfaces[i] = s.strip().replace(' ', '')
    return surfaces

def parse_deltas(file: Path | str, burnin: float) -> list[Delta]:
    deltas = Path(file).open('r').read()
    deltas = deltas.strip().split('\n')[1:]
    deltas = [d.strip().split() for d in deltas]
    deltas = [Delta(int(step), int(index), int(sample), float(nlld))
              for step, index, sample, nlld in deltas]
    deltas = deltas[int(len(deltas) * burnin):]
    return deltas

def parse_forms(file: Path | str) -> dict[int, str]:
    forms = Path(file).open('r').read()
    forms = forms.strip().split('\n')[1:]
    forms = [f.strip().split() for f in forms]
    forms_ = forms
    forms = {}
    for f in forms_:
        if len(f) == 1: f.append('')
        i, f_ = f
        forms[int(i)] = f_
    return forms

def parse_lexicon(file: Path | str) -> Lexicon:
    file = Path(file)
    lines = file.open().read().strip().split('\n')
    lex = {}
    urs = {}
    for line in lines:
        sr, ur = line.strip().split()
        if ur not in urs:
            urs[ur] = len(ur)
        lex[sr] = urs[ur]
    return lex

def get_counts(forms: dict[int, str], deltas: list[Delta]) -> dict[int, Counter[str]]:
    counts = {}
    for d in deltas:
        if d.index not in counts:
            counts[d.index] = Counter()
        counts[d.index][forms[d.sample]] += 1
    return counts

def make_lexicon(surfaces: dict[int, str], counts: dict[int, Counter[str]], out: Path | str) -> Lexicon:
    lex = {}
    urs = {}
    f = Path(out).open('w')
    for i, sr in surfaces.items():
        ur = counts[i].most_common(1)[0][0]
        print(f'{sr}\t{ur}', file=f)
        if ur not in urs:
            urs[ur] = len(ur)
        lex[sr] = urs[ur]
    return lex

def pairs(lex: Lexicon) -> set[tuple[Word]]:
    ps = set()
    for t in set(lex.values()):
        group = {k for k, v in lex.items() if v == t}
        ps |= set(combinations(group, 2))
    return ps

def grouping(refl: Lexicon, hypl: Lexicon):
    refl = {k: v for k, v in refl.items() if k in hypl}
    hypl = {k: v for k, v in hypl.items() if k in refl}
    ref = pairs(refl)
    hyp = pairs(hypl)

    tp = ref & hyp
    try:
        p = len(tp) / len(hyp)
        r = len(tp) / len(ref)
        f = 2 * (p * r) / (p + r)
    except:
        return 0, 0, 0

    return p, r, f

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 5:
        _, surfaces, deltas, forms, gold_lexicon = sys.argv
        deltas = parse_deltas(deltas, 0.5)
        surfaces = parse_surfaces(surfaces)
        forms = parse_forms(forms)
        counts = get_counts(forms, deltas)
        hyp = make_lexicon(surfaces, counts, 'gibbs_lexicon.tsv')
    else:
        _, hyp_lexicon, gold_lexicon = sys.argv
        hyp = parse_lexicon(hyp_lexicon)
    ref = parse_lexicon(gold_lexicon)
    p, r, f = grouping(ref, hyp)

    print(f"Precision: {p}")
    print(f"Recall:    {r}")
    print(f"F-score:   {f}")
    
