from pathlib import Path

def main(phonemes_file, phones_file, out_folder):
    phonemes = Path(phonemes_file).open('r').read().strip().split()
    phones = Path(phones_file).open('r').read().strip().split()
    out = Path(out_folder)

    with open(out / 'naive_universal_grammar.csv', 'w') as naive:
        print('parameter,observation,split', file=naive)
        for phoneme in phonemes:
            for phone in phones:
                print(f'{phoneme},{phone},ug', file=naive)

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

