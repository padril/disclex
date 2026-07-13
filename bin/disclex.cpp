// Copyright (c) Leo Peckham 2026

#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <functional>
#include <utility>
#include <vector>
#include <unordered_map>
#include <deque>
#include <algorithm>

#include "distribution.hpp"
#include "labelling.hpp"
#include "utility.hpp"

#include "impl/semiring.hpp"
using namespace semiring;

#include "external/argparse.hpp"
#include "external/rapidcsv.hpp"
#include "external/expected.hpp"
#include <fst/fstlib.h>
#include <fst/extensions/ngram/ngram-fst.h>
#include <ngram/ngram.h>

// #include "external/ctrack.hpp"

// TODO(padril): badness
using Engine = std::default_random_engine;

#define DISTINCT_SEGMENT(Name) \
    struct Name { \
        int v; \
        Name(int v = 0) : v(v) {} \
        constexpr explicit operator int() const noexcept { return v; } \
        constexpr explicit operator size_t() const noexcept { return v; } \
        constexpr explicit operator bool() const noexcept { return v; } \
        inline constexpr bool operator==(const Name& other) const noexcept { \
            return v == other.v; \
        } \
        inline constexpr bool operator!=(const Name& other) const noexcept { \
            return v != other.v; \
        } \
        inline constexpr bool operator>(const Name& other) const noexcept { \
            return v > other.v; \
        } \
        inline constexpr bool operator<(const Name& other) const noexcept { \
            return v < other.v; \
        } \
        inline constexpr bool operator<=(const Name& other) const noexcept { \
            return v <= other.v; \
        } \
        inline constexpr bool operator>=(const Name& other) const noexcept { \
            return v >= other.v; \
        } \
        bool Write(std::ostream&) const { return false; } \
    }; \
    namespace std { \
        template <> \
        struct hash<Name> { \
            size_t operator()(const Name& s) const noexcept { \
                return std::hash<int>{}(s.v); \
            } \
        }; \
    } \
    std::ostream& operator<<(std::ostream& os, const Name& name) { \
        return os << name.v; \
    }

template <typename T, typename U>
struct VariantSegment {
    int v;
    VariantSegment(int v = 0) : v(v) {}
    VariantSegment(const T& t) : v(t.v) {}
    VariantSegment(const U& u) : v(u.v) {}
    explicit operator T() const noexcept { return v; }
    explicit operator U() const noexcept { return v; }
    constexpr explicit operator int() const noexcept { return v; }
    constexpr explicit operator size_t() const noexcept { return v; }
    inline constexpr bool operator==(const VariantSegment<T, U>& other) const noexcept {
        return v == other.v;
    }
    inline constexpr bool operator!=(const VariantSegment<T, U>& other) const noexcept {
        return v != other.v;
    }
    inline constexpr bool operator>(const VariantSegment<T, U>& other) const noexcept {
        return v > other.v;
    }
    inline constexpr bool operator<(const VariantSegment<T, U>& other) const noexcept {
        return v < other.v;
    }
    inline constexpr bool operator>=(const VariantSegment<T, U>& other) const noexcept {
        return v >= other.v;
    }
    inline constexpr bool operator<=(const VariantSegment<T, U>& other) const noexcept {
        return v <= other.v;
    }
    bool Write(std::ostream&) const { return false; }
};

namespace std {
    template <typename T, typename U>
    struct hash<VariantSegment<T, U>> {
        size_t operator()(const VariantSegment<T, U>& s) const noexcept {
            return std::hash<int>{}(s.v);
        }
    };
}

DISTINCT_SEGMENT(Phoneme);
DISTINCT_SEGMENT(Phone);
DISTINCT_SEGMENT(Aligneme);
using Segment = VariantSegment<Phoneme, Phone>;

// TODO(padril): should we also abstract std::vector to something else?
// using Variable = std::vector<Segment>;  // In the sense of a random variable
using Observation = std::vector<Phone>;
using Parameter = std::vector<Phoneme>;

template <typename Semiring, typename Label>
using FST = fst::VectorFst<fst::ArcTpl<Semiring, Label>>;

static FST<Log, Segment> edit_fst;  // The edit distance FST

template <typename Semiring, typename From>
static RelabelArcMapper<Semiring, From, int> relabel_to_int(
    [](From ilabel, From olabel) -> std::pair<int, int> {
        return std::pair(ilabel.v, olabel.v);
    });
template <typename Semiring, typename To>
static RelabelArcMapper<Semiring, int, To> relabel_from_int(
    [](int ilabel, int olabel) -> std::pair<To, To> {
        return std::pair(To(ilabel), To(olabel));
    });


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

struct Alignment {
    std::vector<std::pair<Phoneme, Phone>> alignment;
    std::string split;
};

enum class Status {
    Invisible,
    Fixed,
    Trainable,
};

struct Schedule {
    double boundary;
    double interior;
    size_t peak_step;
    size_t end_step;
    double at_step(size_t step) {
        if (step <= 0) {
            return boundary;
        } else if (0 < step && step <= peak_step) {
            double delta = (interior - boundary) / peak_step;
            return boundary + delta * step;
        } else if (peak_step < step && step <= end_step) {
            double delta = (interior - boundary) / (end_step - peak_step);
            return interior - delta * (step - peak_step);
        } else {
            return boundary;
        }
    };
};

struct Split {
    Status status;
    Schedule mix;
    size_t order;
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

template <typename Label>
FST<Log, Label> trop_to_log(FST<Trop, Label> trop) {
    FST<Trop, int> trop_int;
    fst::ArcMap(trop, &trop_int, relabel_to_int<Trop, Label>);
    FST<Log, int> log_int;
    fst::ArcMap(trop_int, &log_int, fst::StdToLogMapper());
    FST<Log, Label> log;
    fst::ArcMap(log_int, &log, relabel_from_int<Log, Label>);
    return log;
}

template <typename Label>
FST<Trop, Label> log_to_trop(FST<Log, Label> log) {
    FST<Log, int> log_int;
    fst::ArcMap(log, &log_int, relabel_to_int<Log, Label>);
    FST<Trop, int> trop_int;
    fst::ArcMap(log_int, &trop_int, fst::LogToStdMapper());
    FST<Trop, Label> trop;
    fst::ArcMap(trop_int, &trop, relabel_from_int<Trop, Label>);
    return trop;
}

FST<Log, Segment> count_ngrams(
        std::vector<std::tuple<FST<Log, Phoneme>, FST<Log, Phone>, std::string>> assignments,
        std::unordered_map<std::string, Split> splits,
        Labelling<Aligneme, std::pair<Phoneme, Phone>, std::string>& labelling,
        size_t step,
        bool self_align = false
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
    Aligneme backoff_label = labelling.special("phi");
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

    Segment phi = labelling.decode(backoff_label).first;
    Segment epsilon = labelling.decode(labelling.special("epsilon")).first;
    // Turn <phi>:<phi> transitions to <eps>:<eps>
    // This is the 'Approximate Offline Representation' mentioned in
    // Allauzen et al. 2005.
    // TODO(padril): Implement the 'Exact Offline Representation' as an option
    //               using ngram convert
    // TODO(padril): we can assume a lote more siter and aiter types
    for (fst::StateIterator siter(transform); !siter.Done(); siter.Next()) {
        auto state = siter.Value();
        for (fst::MutableArcIterator aiter(&transform, state);
                !aiter.Done(); aiter.Next()) {
            auto arc = aiter.Value();
            // TODO(padril): should these be static casts everywhere? idk
            if (Phoneme(arc.ilabel) == Phoneme(phi)) {
                arc.ilabel = epsilon;
            }
            if (Phone(arc.olabel) == Phone(phi)) {
                arc.olabel = epsilon;
            }
            aiter.SetValue(arc);
        }
    }

    // TODO(padril): probabilities are not actually normalized right now
    //               because of epsilon issues (or some other rounding
    //               accumulation) I think

    fst::ArcSort(&transform, fst::OLabelCompare<decltype(transform)::Arc>());
    return transform;
}

// TODO(padril): may not need to pass as copy
// TODO(padril): should probably return two FSTs or need to be projected after
template <typename Semiring, template <typename, typename> typename Variant,
          typename ILabel, typename OLabel>
std::pair<FST<Semiring, Variant<ILabel, OLabel>>, Semiring> random_walk(
        Engine& engine, FST<Semiring, Variant<ILabel, OLabel>> fst,
        ILabel istart, OLabel ostart, ILabel iend, OLabel oend
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
    for (fst::ArcIterator aiter(fst, curr); !aiter.Done(); aiter.Next()) {
        auto arc = aiter.Value();
        if ((ILabel) arc.ilabel == istart && (OLabel) arc.olabel == ostart) {
            auto sampled_next = sampled.AddState();
            auto new_arc = typename decltype(sampled)::Arc(
                    arc.ilabel, arc.ilabel, Semiring::One(), sampled_next);
            sampled.AddArc(sampled_curr, new_arc);
            sampled_curr = sampled_next;
            curr = arc.nextstate;
            break;
        }
    }

    // TODO(padril): there is a sampling bug where we get a segfault when we
    //               hit max length
    const size_t  MAX_LENGTH    = 150;

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
    {
        auto sampled_next = sampled.AddState();
        // TODO(padril): all of these auto new_arc's have a better form of Type ident(...);
        auto new_arc = typename decltype(sampled)::Arc(
                iend, oend, Semiring::One(), sampled_next);
        sampled.AddArc(sampled_curr, new_arc);
        sampled_curr = sampled_next;
    }
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

// TODO(padril): could be way better, maybe using boost
struct HashPair {
    template <class T, class U>
    std::size_t operator()(const std::pair<T, U> &p) const {
        auto h1 = std::hash<T>{}(p.first);
        auto h2 = std::hash<U>{}(p.second);
        return h1 ^ h2;
    }
};

// Slightly modified version of the code from @guilhermeagostinelli on github,
// though this is a standard algorithm.
template <typename T, typename U>
int levenshtein(std::vector<T> form1, std::vector<U> form2) {
    int size1 = form1.size();
    int size2 = form2.size();
    std::vector<std::vector<int>> verif(size1 + 1, std::vector<int>(size2 + 1));

    // If one of the words has zero length, the distance is equal to the size of the other word.
    if (size1 == 0) {
        return size2;
    } if (size2 == 0) {
        return size1;
    }

    // Sets the first row and the first column of the verification matrix with the numerical order from 0 to the length of each word.
    for (int i = 0; i <= size1; i++) {
        verif[i][0] = i;
    }
    for (int j = 0; j <= size2; j++) {
        verif[0][j] = j;
    }

    // Verification step / matrix filling.
    for (int i = 1; i <= size1; i++) {
        for (int j = 1; j <= size2; j++) {
            // Sets the modification cost.
            // 0 means no modification (i.e. equal letters) and 1 means that a modification is needed (i.e. unequal letters).
            int cost = (int(form2[j - 1]) == int(form1[i - 1])) ? 0 : 1;

            // Sets the current position of the matrix as the minimum value between a (deletion), b (insertion) and c (substitution).
            // a = the upper adjacent value plus 1: verif[i - 1][j] + 1
            // b = the left adjacent value plus 1: verif[i][j - 1] + 1
            // c = the upper left adjacent value plus the modification cost: verif[i - 1][j - 1] + cost
            verif[i][j] = std::min(
                std::min(verif[i - 1][j] + 1, verif[i][j - 1] + 1),
                verif[i - 1][j - 1] + cost
            );
        }
    }

    // The last position of the matrix will contain the Levenshtein distance.
    return verif[size1][size2];
}


template <typename T, typename Index>
class StrongDeque {
private:
    std::deque<T> data;
public:
    T& operator[](Index i) {
        return data[static_cast<size_t>(i)];
    }
    const T& operator[](Index i) const {
        return data[static_cast<size_t>(i)];
    }
    void push_back(const T& x) {
        data.push_back(x);
    }
    std::size_t size() const {
        return data.size();
    }
};

void gibbs_fst_dijkstra(
        size_t steps, Engine engine,
        std::vector<Alignment> alignments,
        std::unordered_map<std::string, Split> splits,
        Labelling<Aligneme, std::pair<Phoneme, Phone>, std::string>& alignemes,
        Labelling<Phoneme, std::string, std::string>& phonemes,
        Labelling<Phone, std::string, std::string>& phones,
        Log alpha,
        size_t rebuild_every,
        bool self_align,
        double mh_ratio,
        std::ofstream& dp_file,
        std::ofstream& ur_file,
        std::string model_out_dir
        ) {
    enum class ParameterIndex    : size_t {};
    enum class ObservationIndex    : size_t {};
    enum class ModelIndex : size_t {};

    struct Assignment {
        ParameterIndex parameter;
        ObservationIndex observation;
        std::string split;
    };
    
    std::unordered_map<std::string, double> weights;
    for (const auto& [name, split] : splits) {
        weights[name] = split.mix.boundary;
   }

    // TODO(padril): This gets massive. A custom implementation may be better.
    // TODO(padril): switch to unique_ptr across the repo
    // TODO(padril): several of the new FST(...) new UR(...) new SR(...) lines
    //               would be better if we passed an output fst parameter
    // TODO(padril): might be more explanatory to have Segment as just
    //               Variant<Phoneme, Phone>
    StrongDeque<FST<Log, Phoneme>*, ParameterIndex> saved_parameters;
    StrongDeque<FST<Log, Phone>*, ObservationIndex> saved_observations;
    StrongDeque<FST<Log, Segment>*, ModelIndex> saved_models;
    auto transform = ModelIndex(saved_models.size());
    saved_models.push_back(new FST<Log, Segment>());

    std::vector<Assignment> assignments;
    for (const auto& [alignment, split] : alignments) {
        auto [parameter, observation] = unzip(alignment);

        auto parameter_fst = new auto(labels_to_path_fst<Log, Phoneme>(parameter));
        auto parameter_index = ParameterIndex(saved_parameters.size());
        saved_parameters.push_back(parameter_fst);

        auto observation_fst = new auto(labels_to_path_fst<Log, Phone>(observation));
        auto observation_index = ObservationIndex(saved_observations.size());
        saved_observations.push_back(observation_fst);
        
        assignments.push_back(Assignment {
                parameter_index,
                observation_index,
                split,
                });
    }

    // TODO(padril): some noexcepts should be added around the codebase

    distribution::MapValueDistribution uniform(
            new distribution::Uniform<Log::ValueType, Log, Engine>(),
            std::function(real_to_log_semiring)
            );

    // TODO(padril): could be std::tuple<Model, SR, UR>
    std::unordered_map<std::pair<ObservationIndex, ParameterIndex>, Log, HashPair>
        likelihood_cache;
    std::unordered_map<std::pair<ParameterIndex, ModelIndex>, ModelIndex, HashPair>
        parameter_model_compose_cache;
    std::unordered_map<std::pair<ModelIndex, ObservationIndex>, ModelIndex, HashPair>
        model_observation_compose_cache;

    // TODO(padril): these lambdas should probably have explicit captures
    //               but right now, they'd be too large. small refactor.

    distribution::MapGivenDistribution<
        ParameterIndex, Log, Engine, ObservationIndex,
        std::pair<std::optional<ParameterIndex>, std::optional<ObservationIndex>>
        > prior(
        new distribution::FSTDistribution<ParameterIndex, Log, Engine, ModelIndex, ParameterIndex, ObservationIndex>(
            &transform,
            [&](Engine& engine, ModelIndex i) -> std::pair<ParameterIndex, Log> {
                auto [alignment,  p] = random_walk(
                        engine, *saved_models[i],
                        phonemes.special("start"), phones.special("start"),
                        phonemes.special("end"), phones.special("end")
                        );

                auto parameter = new FST<Log, Phoneme>;
                *parameter = project_input(alignment);
                auto parameter_index = ParameterIndex(saved_parameters.size());
                saved_parameters.push_back(parameter);
                return std::pair(parameter_index, p);
            },
            [&](ParameterIndex parameter, ModelIndex model) -> ModelIndex {
                auto it = parameter_model_compose_cache.find(std::pair(parameter, model));
                if (it != parameter_model_compose_cache.end()) {
                    return it->second;
                }
                auto composed = new auto( compose(*saved_parameters[parameter], *saved_models[model]));
                auto composed_index = ModelIndex(saved_models.size());
                saved_models.push_back(composed);
                parameter_model_compose_cache[std::pair(parameter, model)] = composed_index;
                return composed_index;
            },
            [&](ModelIndex model, ObservationIndex observation) -> ModelIndex {
                auto it = model_observation_compose_cache.find(std::pair(model, observation));
                if (it != model_observation_compose_cache.end()) {
                    return it->second;
                }
                auto composed = new auto(compose(*saved_models[model], *saved_observations[observation]));
                auto composed_index = ModelIndex(saved_models.size());
                saved_models.push_back(composed);
                model_observation_compose_cache[std::pair(model, observation)] = composed_index;
                return composed_index;
            }),
        [](ObservationIndex observation) {
            return std::pair(std::nullopt, observation);
        });

    int likelihood_hits;
    int likelihood_misses;

    std::vector<std::pair<ObservationIndex, ParameterIndex>> dp_initialization;
    for (auto [parameter, observation, split] : assignments) {
        if (splits[split].status != Status::Invisible) {
            dp_initialization.push_back(std::pair(observation, parameter));
        }
    }

    distribution::CRPDirichletProcess<ParameterIndex, Log, Engine, ObservationIndex, Log> dp(
            &prior,
            alpha,
            dp_initialization,
            [&](ObservationIndex observation, ParameterIndex parameter) -> Log {
                auto it = likelihood_cache.find(
                        std::pair(observation, parameter));
                if (it != likelihood_cache.end()) {
                    ++likelihood_hits;
                    return it->second;
                }
                ++likelihood_misses;
                auto l = likelihood(*saved_models[transform],
                                    *saved_observations[observation],
                                    *saved_parameters[parameter]);
                likelihood_cache[std::pair(observation, parameter)] = l;
                return l;
            },
            [&](ObservationIndex observation) -> Log {
                // TODO(padril): cache this as well
                FST<Log, Segment>* composed_fst;
                auto it = model_observation_compose_cache.find(
                        std::pair(transform, observation));
                if (it != model_observation_compose_cache.end()) {
                    composed_fst = saved_models[it->second];
                } else {
                    composed_fst = new auto(
                            compose(*saved_models[transform],
                            *saved_observations[observation])
                            );
                    saved_models.push_back(composed_fst);
                    auto composed = ModelIndex(saved_models.size() - 1);
                    model_observation_compose_cache[std::pair(transform, observation)] = composed;
                }
                std::vector<Log> dists;
                // TODO(padril): make shortestdistance gracefully exit when no
                //               dists found
                fst::ShortestDistance(*composed_fst, &dists, true);
                return static_cast<size_t>(composed_fst->Start()) < dists.size() ?
                    dists[composed_fst->Start()] : Log::Zero();
            },
            [&](ObservationIndex observation, ParameterIndex parameter) -> bool {
                Parameter parameter_form = path_fst_to_labels(*saved_parameters[parameter]).value();
                parameter_form.erase(
                        std::remove_if(
                            parameter_form.begin(),
                            parameter_form.end(),
                            [&](Phoneme x) -> bool {
                                return x == phonemes.special("epsilon");
                            }),
                        parameter_form.end());
                Observation observation_form = path_fst_to_labels(*saved_observations[observation]).value();
                observation_form.erase(
                        std::remove_if(
                            observation_form.begin(),
                            observation_form.end(),
                            [&](Phone x) -> bool {
                                return x == phones.special("epsilon");
                            }),
                        observation_form.end());
                // TODO(padril): make this a parameter, and do it better
                Log soft_p = real_to_log_semiring(0.01);
                Log soft_sample = uniform.sample(engine, std::monostate()).first;
                return levenshtein(observation_form, parameter_form) <= 3
                    || soft_sample <= soft_p;
                // retparametern true;  // Default full-conditional behavioparameter
            },
            &uniform);

    // TODO: This can be abstracted, like Uniform, and moved to distributions.hpp
    class UniformProposal :
        public distribution::Distribution<
            ParameterIndex, std::pair<Log, Log>, Engine, std::pair<ParameterIndex, std::pair<ObservationIndex, size_t>>> {
    private:
        distribution::CRPDirichletProcess<
            ParameterIndex, Log, Engine, ObservationIndex, Log>* dp;
    public:
        UniformProposal(
            distribution::CRPDirichletProcess<
                ParameterIndex, Log, Engine, ObservationIndex, Log>* dp
                ) : dp(dp) {};
        std::pair<ParameterIndex, std::pair<Log, Log>> sample(
                Engine& engine,
                std::pair<ParameterIndex, std::pair<ObservationIndex, size_t>> given
                ) {
            ObservationIndex sr = given.second.first;
            std::vector<ParameterIndex> parameters = dp->get_parameters();
            int n = parameters.size();
            std::uniform_int_distribution<int> uniform_pos{0, n};
            int pos = uniform_pos(engine);
            Log p = real_to_log_semiring(1.0 / n);
            if (pos < n) {
                return std::pair(parameters[pos], std::pair(p, p));
            } else {
                return std::pair(
                        std::get<ParameterIndex>(dp->prior->sample(engine, sr)),
                        std::pair(p, p));
            }
            
        }
    };

    // For Metropolis-within-Gibbs
    distribution::MetropolisHastings<ParameterIndex, Log, Engine, std::pair<ObservationIndex, size_t>, Log> mh(
            ParameterIndex(0),
            [&](ParameterIndex parameter, std::pair<ObservationIndex, size_t> observation_count) -> Log {
                auto [observation, count] = observation_count;
                auto it = likelihood_cache.find(std::pair(observation, parameter));
                if (it != likelihood_cache.end()) {
                    ++likelihood_hits;
                    return it->second;
                }
                ++likelihood_misses;
                Log l = likelihood(
                        *saved_models[transform], *saved_observations[observation], *saved_parameters[parameter]);
                likelihood_cache[std::pair(observation, parameter)] = l;
                return real_to_log_semiring(count) * l;
            },
            new UniformProposal(&dp),
            &uniform
            );


    {
        size_t n = assignments.size();
        std::vector<std::tuple<FST<Log, Phoneme>, FST<Log, Phone>,
            std::string>> fst_assignments(n);
        for (size_t i = 0; i < n; ++i) {
            auto [parameter, observation, split] = assignments[i];
            fst_assignments[i] = std::tuple(
                    *saved_parameters[parameter],
                    *saved_observations[observation],
                    split
                    );
        }
        saved_models[transform] = new auto(count_ngrams(fst_assignments, splits, alignemes, 0, self_align));
    }
    std::cout << "\tBuilt initial transform.\n";

    dp_file << "method step i sample nlld\n";
    ur_file << "sample form\n";

    for (size_t step = 0; step < steps; ++step) {
        likelihood_hits = 0;
        likelihood_misses = 0;

        std::cout << "\tStarting step " << step << ".\n";
        // TODO(padril): this is not a good way of doing this
        size_t skipped = 0;
        for (size_t assignment = 0; assignment < assignments.size(); ++assignment) {
            auto [parameter_i, observation_i, split] = assignments[assignment];
            size_t i = assignment - skipped;
            if (splits[split].status != Status::Trainable) {
                ++skipped;
            } else if (uniform.sample(engine, std::monostate()).first < real_to_log_semiring(mh_ratio)) {
                // TODO: this probability is jank
                size_t count = 0;
                std::vector<ParameterIndex> parameters = dp.get_parameters();
                for (ParameterIndex parameter_j : parameters) {
                    if (parameter_i == parameter_j) { ++count; }
                }
                ParameterIndex sample;
                Log p;
                mh.update(parameter_i);
                std::tie(sample, p) = mh.sample(engine, std::pair(observation_i, count));
                if (parameter_i != sample) {
                    dp.update(i, sample);
                    dp_file << "mh " << step << ' ' << i << ' ' << size_t(sample)
                        << ' ' << p.Value() << '\n';
                }
            } else {
                auto [sample, p] = dp.sample(engine, i);
                dp.update(i, sample);
                dp_file << "crp " << step << ' ' << size_t(i) << ' ' << size_t(sample) << ' '
                    << p.Value() << '\n';
            }
        }

        double hit_ratio = static_cast<double>(likelihood_hits)
            / (likelihood_hits + likelihood_misses);
        std::cout << "\t\tLikelihood cache hit ratio : " << hit_ratio << "\n";

        if (step % rebuild_every == rebuild_every - 1) {
            std::cout << "\tRebuilding ngram model\n";
            size_t n = assignments.size();
            std::vector<std::tuple<FST<Log, Phoneme>, FST<Log, Phone>,
                std::string>> fst_assignments(n);
            for (size_t i = 0; i < n; ++i) {
                auto [parameter, observation, split] = assignments[i];
                fst_assignments[i] = std::tuple(
                        *saved_parameters[parameter],
                        *saved_observations[observation],
                        split
                        );
            }
            saved_models[transform] = new auto(
                    count_ngrams(
                        fst_assignments,
                        splits,
                        alignemes,
                        step,
                        self_align
                        ));
            likelihood_cache.clear();
            parameter_model_compose_cache.clear();
            model_observation_compose_cache.clear();

            std::string model_out_path = model_out_dir + "step_"
                + std::to_string(step) + ".csv";
            // TODO(padril): Do not pickle as a .fst, but rather as an
            //               alignment file (which should be standardized for
            //               the repo).
            std::ofstream model_out(model_out_path);
            model_out << "parameter,observation,split\n";
            for (auto [parameter, observation, split] : fst_assignments) {
                auto parameter_labels = path_fst_to_labels(parameter).value();
                auto parameter_string = labels_to_string(parameter_labels, phonemes);
                auto observation_labels = path_fst_to_labels(observation).value();
                auto observation_string = labels_to_string(observation_labels, phones);
                model_out << parameter_string << ","
                    << observation_string << ","
                    << split << "\n";
            };
        }
    }

    for (size_t i = 0; i < saved_parameters.size(); ++i) {
        std::vector<Phoneme> ur = path_fst_to_labels(*saved_parameters[ParameterIndex(i)]).value();
        ur_file << size_t(i) << ' ' << labels_to_string(ur, phonemes) << '\n';
    }
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("disclex");

    // TODO(padril): handle unimplemented geometric prior
    program.add_argument("--prior")
        .choices("geometric", "fst")
        .required()
        .help("which type of prior should be used");

    program.add_argument("--alignments")
        .required()
        .help("the file containing the initial alignment forms. see format "
              "specification in the README");
    program.add_argument("--splits")
        .required()
        .help("the file containing the information about splits. see format "
              "specification in the README");

    // TODO(padril): maybe infer this from the input wordlist
    program.add_argument("--phones")
        .required()
        .help("the file containing the list of phones, each on its own line");

    program.add_argument("--phonemes")
        .required()
        .help("the file containing the list of phonemes, each on its own "
              "line");

    // TODO(padril): handle reading these from the phones file
    program.add_argument("--epsilon")
        .default_value("")
        .help("the string identifying epsilon in phone(me) files, if made "
              "explicit");
    program.add_argument("--phi")
        .default_value("")
        .help("the string identifying phi in phone(me) files, if made "
              "explicit");
    program.add_argument("--start")
        .default_value("")
        .help("the string identifying the start token in phone(me) files, if "
              "made explicit");
    program.add_argument("--end")
        .default_value("")
        .help("the string identifying the end token in phone(me) files, if "
              "made explicit");

    program.add_argument("--geometric-stop-chance")
        .scan<'g', double>()
        .default_value(0.2)
        .help("the stop chance for the geometric prior, if being used");
    // TODO(padril): add to help comment about infinite looping and
    //               possibility of using -1 or --geometric-no-max-length
    program.add_argument("--geometric-max-length")
        .scan<'d', size_t>()
        .default_value(16)
        .help("the maximum length for the geometric prior, if being used");

    program.add_argument("--steps")
        .scan<'d', size_t>()
        .required()
        .help("the number of gibbs steps to do");
    program.add_argument("--alpha")
        .scan<'g', double>()
        .required()
        .help("the alpha parameter for the dirichlet process");
    program.add_argument("--mh-ratio")
        .scan<'g', double>()
        .required()
        .help("the ratio of steps to do as a Metropolis Hastings");

    // should these be FST only?
    program.add_argument("--rebuild-stride")
        .scan<'d', size_t>()
        .required()
        .help("how many steps to take between rebuilding the learned "
              "phonology");
    bool self_align = false;
    program.add_argument("--self-align")
        .store_into(self_align)
        .required()
        .help("what alignment method to use before counting ngrams");

    program.add_argument("--seed")
        .scan<'d', size_t>()
        .help("what seed to use for the RNG if any");

    program.add_argument("--output-deltas")
        .required()
        .help("the file to output DP process deltas to");
    program.add_argument("--output-parameters")
        .required()
        .help("the file to output saved parameters to");
    program.add_argument("--output-models-dir")
        .required()
        .help("the directory to output saved models to");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    Labelling<Phoneme, std::string, std::string> phonemes(
            0,
            [](Phoneme label) -> Phoneme {
                return int(label) + 1;
            });

    Labelling<Phone, std::string, std::string> phones(
            Phone(0),
            [](Phone label) -> Phone {
                return int(label) + 1;
            });

    std::string epsilon_ident = program.get<std::string>("--epsilon");
    phonemes.force_special("epsilon", 0);
    phones.force_special("epsilon", 0);
    if (epsilon_ident != "") {
        phonemes.associate_special("epsilon", epsilon_ident);
        phones.associate_special("epsilon", epsilon_ident);
    } else {
        phonemes.associate_special("epsilon", "<eps>");
        phones.associate_special("epsilon", "<eps>");
    }
    std::string phi_ident = program.get<std::string>("--phi");
    phonemes.special("phi");
    phones.special("phi");
    if (phi_ident != "") {
        phonemes.associate_special("phi", phi_ident);
        phones.associate_special("phi", phi_ident);
    } else {
        phonemes.associate_special("phi", "<phi>");
        phones.associate_special("phi", "<phi>");
    }
    // TODO(padril): add start and end implicitly if they're unset
    std::string start_ident = program.get<std::string>("--start");
    bool explicit_start;
    phonemes.special("start");
    phones.special("start");
    if (start_ident != "") {
        explicit_start = true;
        phonemes.associate_special("start", start_ident);
        phones.associate_special("start", start_ident);
    } else {
        explicit_start = false;
        phonemes.associate_special("start", "<S>");
        phones.associate_special("start", "<S>");
    }
    std::string end_ident = program.get<std::string>("--end");
    bool explicit_end;
    phonemes.special("end");
    phones.special("end");
    if (end_ident != "") {
        explicit_end = true;
        phonemes.associate_special("end", end_ident);
        phones.associate_special("end", end_ident);
    } else {
        explicit_end = false;
        phonemes.associate_special("end", "<E>");
        phones.associate_special("end", "<E>");
    }

    std::string output_deltas_path = program.get<std::string>(
            "--output-deltas");
    std::string output_parameters_path = program.get<std::string>(
            "--output-parameters");
    std::string model_out_dir = program.get<std::string>(
            "--output-models-dir");
    // TODO(padril): safely check that each arg is in a reasonable range
    // TODO(padril): these should probably be unicode capable
    std::string phonemes_path = program.get<std::string>("--phonemes");
    // TODO(padril): safely handle opening files
    std::ifstream phonemes_file(phonemes_path, std::ios_base::in);
    for (std::string line; std::getline(phonemes_file, line);) {
        // TODO(padril): handle whitespace and blank lines
        phonemes.encode(line);
    }
    // TODO(padril): is there a way to defer this?
    phonemes_file.close();

    std::string phones_path = program.get<std::string>("--phones");
    std::ifstream phones_file(phones_path, std::ios_base::in);
    for (std::string line; std::getline(phones_file, line);) {
        // TODO(padril): handle whitespace and blank lines
        phones.encode(line);
    }
    // TODO(padril): is there a way to defer this?
    phones_file.close();

    // Build edit transducer
    {
        auto s = edit_fst.AddState();
        edit_fst.SetStart(s);
        edit_fst.SetFinal(s, Log::One());

        for (Phoneme phoneme : phonemes.labels()) {
            if (phoneme == phonemes.special("phi") ||
                    phoneme == phonemes.special("start") ||
                    phoneme == phonemes.special("end")) { continue; }
            for (Phone phone : phones.labels()) {
                if (phone == phones.special("phi") ||
                        phone == phones.special("start") ||
                        phone == phones.special("end")) { continue; }
                if (phoneme == phonemes.special("epsilon") &&
                        phone == phones.special("epsilon")) { continue; }
                edit_fst.AddArc(s, decltype(edit_fst)::Arc(phoneme, phone, Log::One(), s));
            }
        }

        edit_fst.AddArc(s, decltype(edit_fst)::Arc(
                    phonemes.special("start"), phones.special("start"),
                    Log::One(), s));
        edit_fst.AddArc(s, decltype(edit_fst)::Arc(
                    phonemes.special("end"), phones.special("end"),
                    Log::One(), s));

        fst::ArcSort(&edit_fst, fst::OLabelCompare<decltype(edit_fst)::Arc>());
    }

    std::string alignments_path = program.get<std::string>("--alignments");
    std::string splits_path = program.get<std::string>("--splits");

    size_t steps = program.get<size_t>("--steps");
    double alpha = program.get<double>("--alpha");
    double mh_ratio = program.get<double>("--mh-ratio");

    size_t rebuild_stride = program.get<size_t>("--rebuild-stride");

    // TODO(padril): I don't know if this works
    std::default_random_engine engine;
    if (program.is_used("--seed")) {
        engine.seed(program.get<size_t>("--seed"));
    }

    // MatrixXd transform = MatrixXd::Zero(phones.size(), phonemes.size());
    // for (ulong i = 0; i < phones.size(); ++i) {
    //     for (ulong j = 0; j < phonemes.size(); ++j) {
    //         if (i == j) {
    //             transform(i, j) = 0.9;
    //         } else {
    //             transform(i, j) = 0.1;
    //         }
    //     }
    // }
    // transform(0, 0) = 0.3;

    // parameter,observation,split
    std::vector<Alignment> alignments;
    {
        std::ifstream alignments_file(alignments_path, std::ios_base::in);

        rapidcsv::Document doc(alignments_file, rapidcsv::LabelParams(0, -1));

        std::vector<std::string> observations = doc.GetColumn<std::string>("observation");
        std::vector<std::string> parameters = doc.GetColumn<std::string>("parameter");
        std::vector<std::string> splits = doc.GetColumn<std::string>("split");

        for (size_t i = 0; i < doc.GetRowCount(); ++i) {
            std::vector<Phoneme> ur;
            std::vector<Phone> sr;
            if (!explicit_start) {
                ur.push_back(phonemes.special("start"));
                sr.push_back(phones.special("start"));
            }

            std::istringstream parameter(parameters[i]);
            for (std::string segment; getline(parameter, segment, ' ');) {
                ur.push_back(phonemes.encode(segment));
            }

            std::istringstream observation(observations[i]);
            for (std::string segment; getline(observation, segment, ' ');) {
                sr.push_back(phones.encode(segment));
            }

            if (!explicit_end) {
                ur.push_back(phonemes.special("end"));
                sr.push_back(phones.special("end"));
            }
            // TODO(padril): should error when zip has two args of different length
            alignments.push_back(Alignment {zip(ur, sr), splits[i]});
        }
        std::cout << "Successfully read alignments file.\n";
    }

    // split,status,mix
    std::unordered_map<std::string, Split> splits;
    {
        std::ifstream splits_file(splits_path, std::ios_base::in);

        rapidcsv::Document doc(splits_file, rapidcsv::LabelParams(0, -1));

        std::vector<std::string> names = doc.GetColumn<std::string>("split");
        std::vector<std::string> statuses = doc.GetColumn<std::string>("status");
        std::vector<std::string> mixes = doc.GetColumn<std::string>("mix");
        std::vector<std::string> orders = doc.GetColumn<std::string>("order");

        for (size_t i = 0; i < doc.GetRowCount(); ++i) {
            Status status;
            if (statuses[i] == "invisible") {
                status = Status::Invisible;
            } else if (statuses[i] == "fixed") {
                status = Status::Fixed;
            } else if (statuses[i] == "trainable") {
                status = Status::Trainable;
            } else {
                std::cerr << "Unknown status in initial alignments.\n";
                std::exit(1);
            }

            char colon_sep;
            Schedule mix;
            std::stringstream(mixes[i]) >> mix.boundary
                >> colon_sep >> mix.interior
                >> colon_sep >> mix.peak_step;

            splits[names[i]] = { status, mix, std::stoul(orders[i]) };
        }
        std::cout << "Successfully read splits file.\n";
    }

    Labelling<Aligneme, std::pair<Phoneme, Phone>, std::string> alignemes(
            0,
            [](Aligneme label) -> Aligneme {
                return int(label) + 1;
            });
    // TODO(padril): Move to native FST Symbol maps
    {
        alignemes.force_special("epsilon", 0);
        alignemes.associate_special("epsilon",
                std::pair(phonemes.special("epsilon"), phones.special("epsilon")));
        alignemes.special("phi");
        alignemes.associate_special("phi",
                std::pair(phonemes.special("phi"), phones.special("phi")));
    }
    std::cout << "Starting Gibbs sampling:\n";

    std::ofstream dp_file(output_deltas_path, std::ios_base::out);
    std::ofstream ur_file(output_parameters_path, std::ios_base::out);
    gibbs_fst_dijkstra(
            steps, engine,
            alignments, splits,
            alignemes, phonemes, phones,
            alpha, rebuild_stride, self_align, mh_ratio,
            dp_file, ur_file, model_out_dir
            );

    std::cout << "Completed Gibbs sampling.\n";

    // std::cout << ctrack::result_as_string();
}

