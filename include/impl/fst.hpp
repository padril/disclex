#ifndef FST_HPP_
#define FST_HPP_

#include "fst/fstlib.h"
#include "utility.hpp"
#include "distribution.hpp"
#include "semiring.hpp"
using namespace semiring;

template <typename Semiring, typename Label>
using FST = fst::VectorFst<fst::ArcTpl<Semiring, Label>>;

template <typename Semiring, typename From>
inline RelabelArcMapper<Semiring, From, int> relabel_to_int(
    [](From ilabel, From olabel) -> std::pair<int, int> {
        return std::pair(ilabel.v, olabel.v);
    });
template <typename Semiring, typename To>
inline RelabelArcMapper<Semiring, int, To> relabel_from_int(
    [](int ilabel, int olabel) -> std::pair<To, To> {
        return std::pair(To(ilabel), To(olabel));
    });

// template <typename Semiring, typename From>
// auto relabel_to_int() {
//     return RelabelArcMapper<Semiring, From, int>(
//         [](From ilabel, From olabel) -> std::pair<int, int> {
//             return std::pair(ilabel.v, olabel.v);
//         });
// }
// template <typename Semiring, typename To>
// auto relabel_from_int() {
//     return RelabelArcMapper<Semiring, int, To>(
//         [](int ilabel, int olabel) -> std::pair<To, To> {
//             return std::pair(To(ilabel), To(olabel));
//         });
// }

// TODO(padril): make mutable versions of many of these FST functions
template <typename Semiring, template <typename, typename> typename Variant,
          typename ILabel, typename OLabel>
FST<Semiring, Variant<ILabel, OLabel>> compose(
        const FST<Semiring, Variant<ILabel, OLabel>>& left,
        const FST<Semiring, Variant<ILabel, OLabel>>& right
        ) {
    FST<Semiring, Variant<ILabel, OLabel>> result;
    fst::Compose(left, right, &result);
    return result;
}

template <typename Semiring, template <typename, typename> typename Variant,
          typename ILabel, typename OLabel>
FST<Semiring, Variant<ILabel, OLabel>> compose(
        const FST<Semiring, Variant<ILabel, OLabel>>& left,
        const FST<Semiring, OLabel>& right
        ) {
    FST<Semiring, Variant<ILabel, OLabel>> result;
    fst::Compose(left, cast_output_to_variant<Semiring, Variant, ILabel, OLabel>(right), &result);
    return result;
}

template <typename Semiring, template <typename, typename> typename Variant,
          typename ILabel, typename OLabel>
FST<Semiring, Variant<ILabel, OLabel>> compose(
        const FST<Semiring, ILabel>& left,
        const FST<Semiring, Variant<ILabel, OLabel>>& right
        ) {
    FST<Semiring, Variant<ILabel, OLabel>> result;
    fst::Compose(cast_input_to_variant<Semiring, Variant, ILabel, OLabel>(left), right, &result);
    return result;
}


// TODO(padril): may not need to pass as copy
// TODO(padril): should probably return two FSTs or need to be projected after
template <typename Semiring, template <typename, typename> typename Variant,
          typename ILabel, typename OLabel, typename Engine>
std::pair<FST<Semiring, Variant<ILabel, OLabel>>, Semiring> random_walk(
        Engine& engine, FST<Semiring, Variant<ILabel, OLabel>> fst
        ) {
    // TODO(padril): might want to normalize using forward-backward first
    distribution::MapValueDistribution uniform(
            new distribution::Uniform<typename Semiring::ValueType, Semiring, Engine>(),
            // TODO(padril): Semirings should be more general and have a way
            //               to do this without specifying log semiring
            std::function(real_to_log_semiring)
            );

    FST<Semiring, Variant<ILabel, OLabel>> sampled;
    auto p = Semiring::One();
    auto sampled_curr = sampled.AddState();
    sampled.SetStart(sampled_curr);

    // TODO(padril): fix this being needed to prevent the bug
    fst::RmEpsilon(&fst);

    // fst::RandGen does not return probability weights, so we need to write
    // this ourselves
    auto curr = fst.Start();

    // TODO(padril): there is a sampling bug where we get a segfault when we
    //               hit max length
    const size_t  MAX_LENGTH    = 1024;

    for (size_t n = 0; n < MAX_LENGTH; ++n) {
        // TODO(padril): this is inefficient, could we prenormalize somewhere?
        //               idk
        Semiring final = fst.Final(curr);
        Semiring sum = final;
        for (fst::ArcIterator aiter(fst, curr);
                !aiter.Done(); aiter.Next()) {
            auto arc = aiter.Value();
            sum += arc.weight;
        }
        Semiring bound = final;
        Semiring next = uniform.sample(engine, std::monostate()).first * sum;
        if (next <= bound) {  // Final probability got hit
            p *= bound;
            sampled.SetFinal(sampled_curr, Semiring::One());
            fst::ArcSort(&sampled, fst::ILabelCompare<typename decltype(fst)::Arc>());
            return std::pair(sampled, p);
        }
        if (next <= bound) {  // Final probability got hit
            p *= bound;
            sampled.SetFinal(sampled_curr, Semiring::One());
            fst::ArcSort(&sampled, fst::ILabelCompare<typename decltype(fst)::Arc>());
            return std::pair(sampled, p);
        }
        // Assume that the outgoing arcs are correctly set
        for (fst::ArcIterator aiter(fst, curr); !aiter.Done(); aiter.Next()) {
            auto arc = aiter.Value();
            bound += arc.weight;
            if (next <= bound) {
                auto sampled_next = sampled.AddState();
                // TODO(padril): idk if this should actually be Semiring::One()
                auto new_arc = typename decltype(sampled)::Arc(
                        arc.ilabel, arc.ilabel, Semiring::One(), sampled_next);
                sampled.AddArc(sampled_curr, new_arc);
                p *= arc.weight;
                sampled_curr = sampled_next;
                curr = arc.nextstate;
                break;
            }
        }
    }
    // Hit max length
    sampled.SetFinal(sampled_curr, Semiring::One());
    fst::ArcSort(&sampled, fst::ILabelCompare<typename decltype(sampled)::Arc>());
    return std::pair(sampled, p);
}

template <typename Semiring, template <typename, typename> typename Variant,
          typename ILabel, typename OLabel>
Semiring likelihood(FST<Semiring, Variant<ILabel, OLabel>> model,
                    FST<Semiring, OLabel> observation,
                    FST<Semiring, ILabel> parameter) {
    FST<Semiring, Variant<ILabel, OLabel>> intermediate = compose(model, observation);
    fst::ArcSort(&intermediate, fst::ILabelCompare<typename decltype(intermediate)::Arc>());
    // TODO(padril): I think we want to use the lazy compose for ur o inter
    auto composed = compose(parameter, intermediate);
    std::vector<Semiring> dists;
    fst::ShortestDistance(composed, &dists, true);
    return static_cast<size_t>(composed.Start()) < dists.size() ?
        dists[composed.Start()] : Semiring::Zero();
}

#endif  // define FST_HPP_
