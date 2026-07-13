#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include <fst/fstlib.h>
#include "external/expected.hpp"
#include "labelling.hpp"

template <typename Semiring, typename Label>
using FST = fst::VectorFst<fst::ArcTpl<Semiring, Label>>;

template <typename T, typename U>
constexpr std::vector<std::pair<T, U>> zip(
        std::vector<T> ts, std::vector<U> us) {
    size_t n = std::min(ts.size(), us.size());
    std::vector<std::pair<T, U>> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = std::pair(ts[i], us[i]);
    }
    return result;
}

template <typename T, typename U>
constexpr std::pair<std::vector<T>, std::vector<U>> unzip(
        std::vector<std::pair<T, U>> xs) {
    size_t n = xs.size();
    std::vector<T> ts(n);
    std::vector<U> us(n);
    for (size_t i = 0; i < n; ++i) {
        std::tie(ts[i], us[i]) = xs[i];
    }
    return std::pair(ts, us);
}

template <typename Semiring, template <typename, typename> typename Variant,
          typename ILabel, typename OLabel>
constexpr FST<Semiring, Variant<ILabel, OLabel>> labels_to_path_fst(
        const std::vector<std::pair<ILabel, OLabel>>& labels
        ) noexcept {
    FST<Semiring, Variant<ILabel, OLabel>> result;
    auto state = result.AddState();
    result.SetStart(state);

    for (const auto& [ilabel, olabel] : labels) {
        auto next = result.AddState();
        auto arc = typename decltype(result)::Arc(
                ilabel, olabel, Semiring::One(), next);
        result.AddArc(state, arc);
        state = next;
    }

    result.SetFinal(state, Semiring::One());
    return result;
}

template <typename Semiring, typename Label>
constexpr FST<Semiring, Label> labels_to_path_fst(
        const std::vector<Label>& labels
        ) noexcept {
    FST<Semiring, Label> result;
    auto state = result.AddState();
    result.SetStart(state);

    for (const auto& label : labels) {
        auto next = result.AddState();
        auto arc = typename decltype(result)::Arc(
                label, label, Semiring::One(), next);
        result.AddArc(state, arc);
        state = next;
    }

    result.SetFinal(state, Semiring::One());
    return result;
}

// TODO(padril): explore custom exception types instead of std::string
template <typename Semiring, template <typename, typename> typename Variant,
          typename ILabel, typename OLabel>
constexpr tl::expected<std::vector<std::pair<ILabel, OLabel>>,
                       const std::string_view>
path_fst_to_labels(const FST<Semiring, Variant<ILabel, OLabel>>& path_fst) noexcept {
    if (!path_fst.Properties(fst::kAcyclic, true)) {
        return tl::unexpected("Encountered a cycle in a supposed path FST.");
    }

    std::vector<std::pair<ILabel, OLabel>> labels;

    for (auto state = path_fst.Start(); state != fst::kNoStateId
            && path_fst.Final(state) == Semiring::Zero(); ) {
        fst::ArcIterator<FST<Semiring, Variant<ILabel, OLabel>>> aiter(path_fst, state);

        auto arc = aiter.Value();

        aiter.Next();
        if (!aiter.Done()) {
            return tl::unexpected("Encountered multiple outgoing edges in a supposed path FST.");
        }

        labels.push_back(std::pair((ILabel) arc.ilabel,
                                   (OLabel) arc.olabel));

        state = arc.nextstate;
    }

    return labels;
}

template <typename Semiring, typename Label>
constexpr tl::expected<std::vector<Label>, const std::string_view>
path_fst_to_labels(const FST<Semiring, Label>& path_fst) noexcept {
    if (!path_fst.Properties(fst::kAcyclic, true)) {
        return tl::unexpected("Encountered a cycle in a supposed path FST.");
    }

    std::vector<Label> labels;

    for (auto state = path_fst.Start(); state != fst::kNoStateId
            && path_fst.Final(state) == Semiring::Zero(); ) {
        fst::ArcIterator<FST<Semiring, Label>> aiter(path_fst, state);

        auto arc = aiter.Value();

        aiter.Next();
        if (!aiter.Done()) {
            return tl::unexpected("Encountered multiple outgoing edges in a supposed path FST.");
        }

        if (arc.ilabel != arc.olabel) {
            return tl::unexpected("Differing input and output labels in a non-variant path FST.");
        }

        labels.push_back(arc.ilabel);

        state = arc.nextstate;
    }

    return labels;
}

template <typename Label, typename Special>
std::string labels_to_string(
        const std::vector<Label>& labels,
        const Labelling<Label, std::string, Special>& labelling
        ) noexcept {
    std::string result;
    for (Label label : labels) {
        result += labelling.decode(label);
    }
    return result;
}

template <typename Semiring, typename FromLabel, typename ToLabel>
class RelabelArcMapper {
private:
    using ConvertFn = std::function<
        std::pair<ToLabel, ToLabel>(FromLabel ilabel, FromLabel olabel)>;
    ConvertFn convert;
public: 
    using FromArc = fst::ArcTpl<Semiring, FromLabel>;
    using ToArc = fst::ArcTpl<Semiring, ToLabel>;

    RelabelArcMapper(ConvertFn convert) : convert(convert) {}

    constexpr ToArc operator()(const FromArc &from) const {
        auto [ilabel, olabel] = convert(from.ilabel, from.olabel);
        return ToArc(ilabel, olabel, from.weight, from.nextstate);
    }

    constexpr fst::MapFinalAction FinalAction() const {
        return fst::MAP_NO_SUPERFINAL; 
    }

    constexpr fst::MapSymbolsAction InputSymbolsAction() const {
        return fst::MAP_COPY_SYMBOLS;
    }

    constexpr fst::MapSymbolsAction OutputSymbolsAction() const {
        return fst::MAP_COPY_SYMBOLS;
    }

    constexpr uint64_t Properties(const uint64_t props) const {
        return props;
    }
};

template <typename Semiring, template <typename, typename> typename Variant,
          typename ILabel, typename OLabel>
FST<Semiring, ILabel> project_input(
        FST<Semiring, Variant<ILabel, OLabel>> model
        ) noexcept {
    static RelabelArcMapper<Semiring, Variant<ILabel, OLabel>, ILabel> projection(
            [](Variant<ILabel, OLabel> ilabel, Variant<ILabel, OLabel>
                ) -> std::pair<ILabel, ILabel> {
                return std::pair(ilabel.v, ilabel.v);
            });
    FST<Semiring, ILabel> result;
    fst::ArcMap(model, &result, projection);
    return result;
};

template <typename Semiring, template <typename, typename> typename Variant,
          typename ILabel, typename OLabel>
FST<Semiring, OLabel> project_output(
        FST<Semiring, Variant<ILabel, OLabel>> model
        ) noexcept {
    static RelabelArcMapper<Semiring, Variant<ILabel, OLabel>, ILabel> projection(
            [](Variant<ILabel, OLabel>, Variant<ILabel, OLabel> olabel
                ) -> std::pair<OLabel, OLabel> {
                return std::pair(olabel.v, olabel.v);
            });
    FST<Semiring, OLabel> result;
    fst::ArcMap(model, &result, projection);
    return result;
};

template <typename Semiring, template <typename, typename> typename Variant,
          typename ILabel, typename OLabel>
FST<Semiring, Variant<ILabel, OLabel>> cast_input_to_variant(
        FST<Semiring, ILabel> model
        ) noexcept {
    static RelabelArcMapper<Semiring, ILabel, Variant<ILabel, OLabel>> projection(
            [](ILabel ilabel, ILabel
                ) -> std::pair<Variant<ILabel, OLabel>, Variant<ILabel, OLabel>> {
                return std::pair(ilabel, ilabel);
            });
    FST<Semiring, Variant<ILabel, OLabel>> result;
    fst::ArcMap(model, &result, projection);
    return result;
};

template <typename Semiring, template <typename, typename> typename Variant,
          typename ILabel, typename OLabel>
FST<Semiring, Variant<ILabel, OLabel>> cast_output_to_variant(
        FST<Semiring, OLabel> model
        ) noexcept {
    static RelabelArcMapper<Semiring, OLabel, Variant<ILabel, OLabel>> projection(
            [](OLabel, OLabel olabel
                ) -> std::pair<Variant<ILabel, OLabel>, Variant<ILabel, OLabel>> {
                return std::pair(olabel, olabel);
            });
    FST<Semiring, Variant<ILabel, OLabel>> result;
    fst::ArcMap(model, &result, projection);
    return result;
};

#endif  // UTILITY_H


