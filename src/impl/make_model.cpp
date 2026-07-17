#include "impl/make_model.hpp"
#include <ngram/ngram.h>

double Schedule::at_step(size_t step)  {
    // Start and end values being near zero prevents crashes in some cases
    double start_value = 0.0001;
    size_t start_step = 0;
    double end_value = 0.0001;
    size_t end_step = max_step;
    for (auto [point_value, point_step] : points) {
        if (step == point_step) {
            return point_value;
        } else if (step <= point_step && point_step <= end_step) {
            end_value = point_value;
            end_step = point_step;
        } else if (start_step <= point_step && point_step <= step) {
            start_value = point_value;
            start_step = point_step;
        }
    }
    auto delta_step = end_step - start_step;
    auto delta_value = end_value - start_value;
    auto slope = delta_value / delta_step;
    return start_value + slope * (step - start_step);
};


FST<Trop, int> ngrams_from_union(FST<Log, Aligneme> union_, Split split, Aligneme backoff_label) {
    // The API requires that labels are ints for the purpose of counting
    FST<Log, int> union_int;
    fst::ArcMap(union_, &union_int, relabel_to_int<Log, Aligneme>);
    fst::Determinize(union_int, &union_int);
    fst::RmEpsilon(&union_int);
    auto [mix, schedule, order] = split;
    ngram::NGramCounter<Trop, int> counter(order);
    counter.Count(union_int);

    FST<Trop, int> model_int;
    counter.GetFst(&model_int);

    // TODO(padril): decltype(...) a LOT more throughout the codebase
    fst::ArcSort(&model_int, fst::ILabelCompare<decltype(model_int)::Arc>());
    ngram::NGramKneserNey make(&model_int, false, int(backoff_label));
    // TODO(padril): look at the other options of this
    // TODO(padril): maybe use ngram shrink
    if (!make.MakeNGramModel()) {
        std::cerr << "Failed to make NGram model\n";
    }
    return model_int;
}

FST<Log, Segment> count_ngrams(
        std::vector<std::tuple<FST<Log, Phoneme>, FST<Log, Phone>, std::string>> assignments,
        std::unordered_map<std::string, Split> splits,
        Labelling<Aligneme, std::pair<Phoneme, Phone>, std::string>& labelling,
        FST<Log, Segment> edit_fst,
        size_t step,
        bool self_align
        ) {
    // TODO(padril): Is this function actually correct?
    //  - We might not be weighting more common alignments better
    RelabelArcMapper<Log, Segment, Aligneme> transducer_to_acceptor(
            [&labelling](Segment ilabel, Segment olabel
                ) -> std::pair<Aligneme, Aligneme> {
                Aligneme aligneme = labelling.encode(
                        std::pair(Phoneme(ilabel), Phone(olabel)));
                return std::pair(aligneme, aligneme);
            });
    RelabelArcMapper<Trop, Aligneme, Segment> acceptor_to_transducer(
            [&labelling](Aligneme aligneme, Aligneme
                ) -> std::pair<Segment, Segment> {
                // TODO(padril): no check, but aligneme ilabel should eq olabel
                return labelling.decode(aligneme);
            });
    RelabelArcMapper<Log, Phoneme, Segment> phoneme_to_segment(
            [](Phoneme ilabel, Phoneme olabel) -> std::pair<Segment, Segment> {
                return std::pair(ilabel, olabel);
            });

    std::unordered_map<std::string, FST<Log, Aligneme>> unions;

    // TODO(padril): the invisible models should be generated ahead of time
    for (const auto& [parameter, observation, split] : assignments) {
        FST<Log, Segment> alignment;
        if (self_align) {
            fst::ArcMap(parameter, &alignment, phoneme_to_segment);
        } else {
            auto intermediate = compose(edit_fst, observation);
            fst::ArcSort(&intermediate,
                    fst::ILabelCompare<decltype(intermediate)::Arc>());
            auto possible_alignments = compose(parameter, intermediate);

            FST<Trop, Segment> trop_possible_alignments = log_to_trop(possible_alignments);
            FST<Trop, Segment> trop_alignment;
            fst::ShortestPath(trop_possible_alignments, &trop_alignment);
            alignment = trop_to_log(trop_alignment);
        }
        FST<Log, Aligneme> alignment_acceptor;
        fst::ArcMap(alignment, &alignment_acceptor, transducer_to_acceptor);
        fst::Union(&unions[split], alignment_acceptor);
    }


    // TODO(padril): look into other smoothing options again
    // TODO(padril): look into .AddSequence
    double base_mix;
    FST<Trop, int> acceptor_int;
    Aligneme backoff_label = labelling.special("epsilon");
    {
        auto max = std::max_element(
                unions.begin(), unions.end(),
                [&](auto l, auto r) -> bool {
                    auto [lsplit, lunion] = l;
                    auto [rsplit, runion] = r;
                    return splits[lsplit].order < splits[rsplit].order;
                });
        auto [split, union_] = *max;
        unions.erase(max);
        acceptor_int = ngrams_from_union(union_, splits[split], backoff_label);
        base_mix = splits[split].mix.at_step(step);
    }
    ngram::NGramModelMerge merger(&acceptor_int, int(backoff_label));
    for (auto [split, union_] : unions) {
        auto model_int = ngrams_from_union(union_, splits[split], backoff_label);
        // TODO(padril): check if mixing rations are in the right order
        auto mix = splits[split].mix.at_step(step);
        merger.MergeNGramModels(model_int, base_mix, mix, true);
        base_mix = 1.0;
    }

    FST<Trop, Aligneme> acceptor;
    fst::ArcMap(acceptor_int, &acceptor, relabel_from_int<Trop, Aligneme>);
    FST<Trop, Segment> transform_trop;
    fst::ArcMap(acceptor, &transform_trop, acceptor_to_transducer);
    FST<Log, Segment> transform = trop_to_log(transform_trop);

    // Segment phi = labelling.decode(backoff_label).first;
    // Segment epsilon = labelling.decode(labelling.special("epsilon")).first;
    // // Turn <phi>:<phi> transitions to <eps>:<eps>
    // // This is the 'Approximate Offline Representation' mentioned in
    // // Allauzen et al. 2005.
    // // TODO(padril): Implement the 'Exact Offline Representation' as an option
    // //               using ngram convert
    // // TODO(padril): we can assume a lote more siter and aiter types
    // for (fst::StateIterator siter(transform); !siter.Done(); siter.Next()) {
    //     auto state = siter.Value();
    //     for (fst::MutableArcIterator aiter(&transform, state);
    //             !aiter.Done(); aiter.Next()) {
    //         auto arc = aiter.Value();
    //         // TODO(padril): should these be static casts everywhere? idk
    //         if (Phoneme(arc.ilabel) == Phoneme(phi)) {
    //             arc.ilabel = epsilon;
    //         }
    //         if (Phone(arc.olabel) == Phone(phi)) {
    //             arc.olabel = epsilon;
    //         }
    //         aiter.SetValue(arc);
    //     }
    // }

    // TODO(padril): probabilities are not actually normalized right now
    //               because of epsilon issues (or some other rounding
    //               accumulation) I think

    fst::ArcSort(&transform, fst::OLabelCompare<decltype(transform)::Arc>());
    return transform;
}
