./build/bin/disclex \
    --prior fst \
    --alignments data/splits/train/alignments.csv \
    --alignments data/splits/train/ident.csv \
    --alignments data/naive_universal_grammar.csv \
    --splits data/conditions/training_splits.txt \
    --phones data/segments.txt \
    --phonemes data/segments.txt \
    --start '<S>' \
    --end '<E>' \
    --steps 2000 \
    --alpha 8 \
    --rebuild-stride 50 \
    --mh-ratio 0 \
    --output-deltas out/training/deltas.out \
    --output-parameters out/training/parameters.out \
    --output-models-dir out/training/models/


