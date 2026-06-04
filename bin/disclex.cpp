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

#include "distribution.hpp"
#include "labelling.hpp"

#include "external/argparse.hpp"
#include <fst/fstlib.h>
#include <ngram/ngram.h>

// #include "external/ctrack.hpp"

// using Eigen::MatrixXd;

using Label = int;
using Char = char;
using Text = std::string;
using IFStream = std::basic_ifstream<Char>;
using Semiring = fst::LogWeightTpl<float>;
using Engine = std::default_random_engine;
using Arc = fst::ArcTpl<Semiring, Label>;
template <typename A>
using FSTType = fst::VectorFst<A>;
using FST = FSTType<Arc>;
using NGramFST = fst::StdVectorFst;
using Form = std::vector<Label>;
using SR = FST;
using UR = FST;
using AR = FST;
template <typename Index, typename Held>
using LexiconMap = std::vector<Held>;
using Lexicon = LexiconMap<int, std::pair<UR, Semiring>>;


// Doing a lot of operator overloading allows the API to behave better when
// we use the Semirings as we would doubles
namespace fst {

Semiring& operator+=(Semiring& a, Semiring b) {
    a = fst::Plus(a, b);
    return a;
}

Semiring operator+(Semiring a, Semiring b) {
    a += b;
    return a;
}

Semiring& operator-=(Semiring& a, Semiring b) {
    a = fst::Minus(a, b);
    return a;
}

Semiring operator-(Semiring a, Semiring b) {
    a -= b;
    return a;
}

Semiring& operator*=(Semiring& a, Semiring b) {
    a = fst::Times(a, b);
    return a;
}

Semiring operator*(Semiring a, Semiring b) {
    a *= b;
    return a;
}

Semiring& operator/=(Semiring& a, Semiring b) {
    a = fst::Divide(a, b);
    return a;
}

Semiring operator/(Semiring a, Semiring b) {
    a /= b;
    return a;
}

bool operator<(Semiring a, Semiring b) {
    return a.Value() > b.Value();
}

bool operator<=(Semiring a, Semiring b) {
    return a < b || a == b;
}

bool operator>(Semiring a, Semiring b) {
    return !(a < b);
}

bool operator>=(Semiring a, Semiring b) {
    return a > b || a == b;
}

};  // namespace fst

class UniformSemiring : public distribution::Distribution<
                        Semiring, std::monostate, Engine, std::monostate> {
 private:
    std::uniform_real_distribution<Semiring::ValueType> uniform{0.0, 1.0};
 public:
    UniformSemiring() {}
    std::pair<Semiring, std::monostate> sample(
            Engine& engine, std::monostate) override {
        return std::pair(Semiring(-std::log(uniform(engine))),
                std::monostate());
    }
};

// TODO(padril): std library or boost might have something for this?
std::vector<Text> split(const Text& s, Char delim) {
    std::vector<Text> out;
    std::size_t start = 0;

    while (true) {
        auto pos = s.find(delim, start);
        if (pos == Text::npos) {
            out.emplace_back(s.substr(start));
            break;
        }
        out.emplace_back(s.substr(start, pos - start));
        start = pos + 1;
    }

    return out;
}

FST form_to_fst(Form form) {
    FST fst;
    auto state = fst.AddState();
    fst.SetStart(state);

    for (Label phon : form) {
        auto next = fst.AddState();
        fst.AddArc(state, Arc(phon, phon, Semiring::One(), next));
        state = next;
    }

    fst.SetFinal(state, Semiring::One());
    return fst;
}

FST align_to_fst(Form ur, Form sr) {
    FST ar;
    auto state = ar.AddState();
    ar.SetStart(state);

    for (size_t i = 0; i < ur.size(); ++i) {
        auto next = ar.AddState();
        ar.AddArc(state, Arc(ur[i], sr[i], Semiring::One(), next));
        state = next;
    }

    ar.SetFinal(state, Semiring::One());
    return ar;
}

Form fst_to_form(const FST& fst) {
    Form form;

    // fst::StdVectorFst trop_fst;
    // fst::StdVectorFst best;
    // fst::ArcMap(log_fst, &trop_fst, fst::LogToStdMapper());
    // fst::ShortestPath(trop_fst, &best);

    auto state = fst.Start();
    if (state == fst::kNoStateId) return form;

    // TODO(padril): This should return something different if the FST is
    //               non-linear
    // TODO(padril): arbitrary max_length
    for (size_t i = 0; i < 30; ++i) {
        if (fst.Final(state) != Semiring::Zero()) {
            break;
        }

        fst::ArcIterator<FST> aiter(fst, state);

        Arc arc = aiter.Value();
        aiter.Next();

        form.push_back(arc.ilabel);
        state = arc.nextstate;
    }

    return form;
}


std::string display_form(
        Form form, Labelling<Label, Text, Text>& labelling,
        std::string lwrap = "",
        std::string rwrap = "",
        std::string end = "\n") {
    std::string ret;
    ret += lwrap;
    for (Label segment : form) {
        if (segment == labelling.special("epsilon")) {
            continue;
        }
        ret += labelling.decode(segment);
    }
    ret += rwrap + end;
    return ret;
}

template <typename A>
fst::VectorFst<A> to_acceptor(
        Labelling<Label, std::pair<Label, Label>, Text>& labelling,
        fst::VectorFst<A> fst) {
    fst::VectorFst<A> result;

    for (fst::StateIterator<fst::VectorFst<A>> siter(fst);
            !siter.Done(); siter.Next()) {
        result.AddState();
    }

    result.SetStart(fst.Start());

    for (fst::StateIterator<fst::VectorFst<A>> siter(fst);
            !siter.Done(); siter.Next()) {
        auto s = siter.Value();
        if (fst.Final(s) != A::Weight::Zero()) {
            result.SetFinal(s, A::Weight::One());  // keep weight 0, adjust if
                                                   // needed
        }

        for (fst::ArcIterator<fst::VectorFst<A>> aiter(fst, s);
                !aiter.Done(); aiter.Next()) {
            auto a = aiter.Value();
            Label label = labelling.encode(std::pair(a.ilabel, a.olabel));
            result.AddArc(s, A(label, label, a.weight, a.nextstate));
        }
    }

    return result;
}

template <typename A>
fst::VectorFst<A> from_acceptor(
        Labelling<Label, std::pair<Label, Label>, Text>& labelling,
        fst::VectorFst<A> fst) {
    fst::VectorFst<A> result;

    for (fst::StateIterator<fst::VectorFst<A>> siter(fst);
            !siter.Done(); siter.Next()) {
        result.AddState();
    }

    result.SetStart(fst.Start());

    for (fst::StateIterator<fst::VectorFst<A>> siter(fst);
            !siter.Done(); siter.Next()) {
        auto s = siter.Value();
        if (fst.Final(s) != A::Weight::Zero()) {
            result.SetFinal(s, A::Weight::One());
        }

        for (fst::ArcIterator<fst::VectorFst<A>> aiter(fst, s);
                !aiter.Done(); aiter.Next()) {
            auto a = aiter.Value();
            auto pair = labelling.decode(a.ilabel);
            auto new_arc = A(pair.first, pair.second, a.weight, a.nextstate);
            result.AddArc(s, new_arc);
        }
    }

    return result;
}

FST ngram_counts(
        std::vector<UR> parameters, NGramFST ug_model,
        Labelling<Label, std::pair<Label, Label>, Text>& labelling,
        double ug_weight) {
    FST top;

    for (UR ur : parameters) {
        fst::Union(&top, to_acceptor(labelling, ur));
    }
    fst::RmEpsilon(&top);
    fst::Determinize(top, &top);

    // TODO(padril): generalize order
    // TODO(padril): look into .AddSequence
    ngram::NGramCounter<NGramFST::Arc::Weight, Label> counter(2);
    if (!counter.Count(top)) {
        std::cerr << "Count n-gram FST could not be properly computed\n";
    }

    NGramFST model;
    counter.GetFst(&model);

    // Must be one of:
    // - kneser_ney
    // - absolute
    // - katz
    // - witten_bell
    // - unsmoothed
    // - presmoothed
    // TODO(padril): allow other smoothing methods
    std::string method = "kneser_ney";

    Label backoff_label = labelling.special("phi");
    fst::ArcSort(&model, fst::ILabelCompare<NGramFST::Arc>());
    ngram::NGramKneserNey make(&model, false, backoff_label);
    // TODO(padril): look at the other options of this
    // TODO(padril): maybe use ngram shrink
    make.MakeNGramModel();

    ngram::NGramModelMerge merger(&model, backoff_label);
    // TODO(padril): check if alpha is in the right order
    merger.MergeNGramModels(ug_model, 1.0, ug_weight, true);

    FST transform;
    fst::ArcMap(model, &transform, fst::StdToLogMapper());
    transform = from_acceptor(labelling, transform);

    Label phi = labelling.decode(backoff_label).first;
    Label epsilon_epsilon = labelling.special("epsilon");
    Label epsilon = labelling.decode(epsilon_epsilon).first;
    // Turn <phi>:<phi> transitions to <eps>:<eps>
    // This is the 'Approximate Offline Representation' mentioned in
    // Allauzen et al. 2005.
    // TODO(padril): Implement the 'Exact Offline Representation' as an option
    //               using ngram convert
    for (fst::StateIterator<FST> siter(transform);
            !siter.Done(); siter.Next()) {
        FST::StateId state = siter.Value();
        for (fst::MutableArcIterator<FST> aiter(&transform, state);
                !aiter.Done(); aiter.Next()) {
            Arc arc = aiter.Value();
            if (arc.ilabel == phi) {
                arc.ilabel = epsilon;
            }
            if (arc.olabel == phi) {
                arc.olabel = epsilon;
            }
            aiter.SetValue(arc);
        }
    }

    // TODO(padril): probabilities are not actually normalized right now
    //               because of epsilon issues (or some other rounding
    //               accumulation) I think

    fst::ArcSort(&transform, fst::OLabelCompare<Arc>());
    return transform;
}

// TODO(padril): may not need to pass as copy
std::pair<FST, Semiring> random_walk(Engine& engine, FST fst) {
    // TODO(padril): might want to normalize using forward-backward first
    UniformSemiring uniform;

    UR ur;
    Semiring p = Semiring::One();
    FST::StateId ur_curr = ur.AddState();
    ur.SetStart(ur_curr);

    // TODO(padril): fix this being needed to prevent the bug
    fst::RmEpsilon(&fst);

    // fst::RandGen does not return probability weights, so we need to write
    // this ourselves
    FST::StateId curr = fst.Start();

    // TODO(padril): there is a sampling bug where we get a segfault when we
    //               hit max length
    const size_t  MAX_LENGTH    = 150;

    for (size_t n = 0; n < MAX_LENGTH; ++n) {
        // TODO(padril): this is inefficient, could we prenormalize somewhere?
        //               idk
        Semiring final = fst.Final(curr);
        Semiring sum = final;
        for (fst::ArcIterator<FST> aiter(fst, curr);
                !aiter.Done(); aiter.Next()) {
            Arc arc = aiter.Value();
            sum += arc.weight;
        }
        Semiring bound = final;
        Semiring next = std::get<Semiring>(
                uniform.sample(engine, std::monostate())) * sum;
        if (next <= bound) {  // Final probability got hit
            p *= bound;
            ur.SetFinal(ur_curr, Semiring::One());
            fst::ArcSort(&ur, fst::ILabelCompare<Arc>());
            return std::pair(ur, p);
        }
        // Assume that the outgoing arcs are correctly set
        for (fst::ArcIterator<FST> aiter(fst, curr);
                !aiter.Done(); aiter.Next()) {
            Arc arc = aiter.Value();
            bound += arc.weight;
            if (next <= bound) {
                FST::StateId ur_next = ur.AddState();
                // TODO(padril): idk if this should actually be Semiring::One()
                auto new_arc = Arc(
                        arc.ilabel, arc.ilabel, Semiring::One(), ur_next);
                ur.AddArc(ur_curr, new_arc);
                p *= arc.weight;
                ur_curr = ur_next;
                curr = arc.nextstate;
                break;
            }
        }
    }
    // Hit max length
    ur.SetFinal(ur_curr, Semiring::One());
    fst::ArcSort(&ur, fst::ILabelCompare<Arc>());
    return std::pair(ur, p);
}

inline
FST compose(const FST& left, const FST& right) {
    FST ret;
    fst::Compose(left, right, &ret);

    return ret;
}

Semiring likelihood(FST fst, SR sr, UR ur) {
    FST intermediate = compose(fst, sr);
    fst::ArcSort(&intermediate, fst::ILabelCompare<Arc>());
    FST composed = compose(ur, intermediate);
    std::vector<Semiring> dists;
    fst::ShortestDistance(composed, &dists, true);
    return dists[composed.Start()];
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
int levenshtein(Form form1, Form form2) {
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
            int cost = (form2[j - 1] == form1[i - 1]) ? 0 : 1;

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

// TODO(padril): split Form into UR and SR forms?
// and maybe better name SR and UR
void gibbs_fst_dijkstra(
        int steps, Engine engine, std::vector<Form> surface_forms,
        NGramFST ug_counts,
        Labelling<Label, std::pair<Label, Label>, Text>& alignemes,
        Labelling<Label, Text, Text>& phonemes,
        Semiring alpha,
        double ug_weight,
        size_t rebuild_every,
        std::ofstream& dp_file,
        std::ofstream& ur_file,
        std::string fst_out_dir) {
    using Index = int;

    // TODO(padril): This gets massive. A custom implementation may be better.
    // TODO(padril): switch to unique_ptr across the repo
    // TODO(padril): several of the new FST(...) new UR(...) new SR(...) lines
    //               would be better if we passed an output fst parameter
    std::deque<FST*> saved_fsts;

    saved_fsts.push_back(new FST());
    Index ngram = saved_fsts.size() - 1;

    // TODO(padril): Lexicon should be its own abstraction
    std::vector<Index> observations;
    std::vector<Index> initial_parameters;
    for (Form sr : surface_forms) {
        UR* ur_fst = new UR(form_to_fst(sr));
        // UR* ur_fst = new UR(form_to_fst(Form({
        //                 phonemes.special("start"),
        //                 phonemes.special("end")
        //                 })));
        SR* sr_fst = new SR(form_to_fst(sr));
        saved_fsts.push_back(ur_fst);
        initial_parameters.push_back(saved_fsts.size() - 1);
        saved_fsts.push_back(sr_fst);
        observations.push_back(saved_fsts.size() - 1);
    }
    std::cout << "\tCompleted initial alignment.\n";

    UniformSemiring uniform;

    std::unordered_map<std::pair<Index, Index>, Semiring, HashPair>
        likelihood_cache;
    std::unordered_map<std::pair<Index, Index>, Index, HashPair>
        compose_cache;

    distribution::MapGivenDistribution<
        Index, Semiring, Engine, Index,
        std::pair<std::optional<Index>, std::optional<Index>>> prior(
        new distribution::FSTDistribution<Index, Semiring, Engine, Index>(
            &ngram,
            [&saved_fsts](
                Engine& engine, Index fst) -> std::pair<Index, Semiring> {
                UR* ur = new UR();
                Semiring p;
                std::tie(*ur, p) = random_walk(engine, *saved_fsts[fst]);
                saved_fsts.push_back(ur);
                return std::pair(saved_fsts.size() - 1, p);
            },
            [&saved_fsts, &compose_cache](Index left, Index right) -> Index {
                auto it = compose_cache.find(std::pair(left, right));
                if (it != compose_cache.end()) {
                    return it->second;
                }
                FST* composed_fst = new FST(
                        compose(*saved_fsts[left], *saved_fsts[right]));
                saved_fsts.push_back(composed_fst);
                Index composed = saved_fsts.size() - 1;
                compose_cache[std::pair(left, right)] = composed;
                return composed;
            }),
        [](Index sr) {
            return std::pair(std::nullopt, sr);
        });

    int likelihood_hits;
    int likelihood_misses;

    // TODO(padril): LexiconMap is poorly named
    distribution::CRPDirichletProcess<
        Index, Semiring, Engine, Index, Index, LexiconMap> dp(
            &prior,
            alpha,
            observations,
            initial_parameters,
            [&saved_fsts, &likelihood_cache, &ngram, &likelihood_hits, &likelihood_misses](
                Index sr, Index ur) -> Semiring {
                auto it = likelihood_cache.find(std::pair(sr, ur));
                if (it != likelihood_cache.end()) {
                    ++likelihood_hits;
                    return it->second;
                }
                ++likelihood_misses;
                Semiring l = likelihood(
                        *saved_fsts[ngram], *saved_fsts[sr], *saved_fsts[ur]);
                likelihood_cache[std::pair(sr, ur)] = l;
                return l;
            },
            [&saved_fsts, &compose_cache, &ngram](Index sr) -> Semiring {
                // TODO(padril): cache this as well
                FST* composed_fst;
                auto it = compose_cache.find(std::pair(ngram, sr));
                if (it != compose_cache.end()) {
                    composed_fst = saved_fsts[it->second];
                } else {
                    composed_fst = new FST(
                            compose(*saved_fsts[ngram], *saved_fsts[sr]));
                    saved_fsts.push_back(composed_fst);
                    Index composed = saved_fsts.size() - 1;
                    compose_cache[std::pair(ngram, sr)] = composed;
                }
                std::vector<Semiring> dists;
                // TODO(padril): make shortestdistance gracefully exit when no
                //               dists found
                fst::ShortestDistance(*composed_fst, &dists, true);
                return dists[composed_fst->Start()];
            },
            [&saved_fsts](Index sr, Index ur) -> bool {
                Form sr_form = fst_to_form(*saved_fsts[sr]);
                Form ur_form = fst_to_form(*saved_fsts[ur]);
                return levenshtein(sr_form, ur_form) <= 3;
                // return true;  // Default full-conditional behaviour
            },
            &uniform);

    std::vector<UR> parameters;
    for (Index i : dp.get_parameters()) {
        parameters.push_back(*saved_fsts[i]);
    }

    // TODO: This can be abstracted, like Uniform, and moved to distributions.hpp
    class UniformProposal :
        public distribution::Distribution<
            Index, std::pair<Semiring, Semiring>, Engine, Index> {
    private:
        distribution::CRPDirichletProcess<
            Index, Semiring, Engine, Index, Index, LexiconMap>* dp;
    public:
        UniformProposal(
            distribution::CRPDirichletProcess<
                Index, Semiring, Engine, Index, Index, LexiconMap>* dp
                ) : dp(dp) {};
        std::pair<Index, std::pair<Semiring, Semiring>> sample(Engine& engine, Index given) {
            LexiconMap<int, Index> parameters = dp->get_parameters();
            int n = parameters.size();
            std::uniform_int_distribution<int> uniform_pos{0, n};
            int pos = uniform_pos(engine);
            Semiring p = Semiring(-std::log(1.0 / n));
            if (pos < n) {
                return std::pair(parameters[pos], std::pair(p, p));
            } else {
                return std::pair(
                        std::get<Index>(dp->prior->sample(engine, given)),
                        std::pair(p, p));
            }
            
        }
    };

    // For Metropolis-within-Gibbs
    distribution::MetropolisHastings<Index, Semiring, Engine, std::pair<Index, int>> mh(
            0,
            [&saved_fsts, &likelihood_cache, &ngram, &likelihood_hits, &likelihood_misses](
                Index ur, std::pair<Index, int> sr_count) -> Semiring {
                Index sr;
                int count;
                std::tie(sr, count) = sr_count;
                auto it = likelihood_cache.find(std::pair(sr, ur));
                if (it != likelihood_cache.end()) {
                    ++likelihood_hits;
                    return it->second;
                }
                ++likelihood_misses;
                Semiring l = likelihood(
                        *saved_fsts[ngram], *saved_fsts[sr], *saved_fsts[ur]);
                likelihood_cache[std::pair(sr, ur)] = l;
                return Semiring(-std::log(count)) * l;
            },
            new UniformProposal(&dp),
            &uniform
            );


    saved_fsts[ngram] = new FST(
            ngram_counts(parameters, ug_counts, alignemes, ug_weight));
    std::cout << "\tCompleted initial NGram count.\n";

    dp_file << "method step i sample nlld\n";
    ur_file << "sample form\n";

    for (int step = 0; step < steps; ++step) {
        likelihood_hits = 0;
        likelihood_misses = 0;

        std::cout << "\tStarting step " << step << ".\n";

        for (size_t i = 0; i < observations.size(); ++i) {
            // TODO: Jank
            // if (std::get<Semiring>(uniform.sample(engine, std::monostate())) < Semiring(-std::log(0.1))) {
            if (true) {
                Index sample;
                Semiring sample_p;
                std::tie(sample, sample_p) = dp.sample(engine, i);
                dp.update(i, sample);
                dp_file << "crp " << step << ' ' << i << ' ' << sample << ' '
                    << sample_p.Value() << '\n';
            } else {
                int count = 0;
                LexiconMap<int, Index> parameters = dp.get_parameters();
                for (Index parameter_j : parameters) {
                    if (parameters[i] == parameter_j) { ++count; }
                }
                Index sample;
                Semiring sample_p;
                mh.update(parameters[i]);
                std::tie(sample, sample_p) = mh.sample(engine, std::pair(i, count));
                if (parameters[i] != sample) {
                    dp.update(i, sample);
                    dp_file << "mh " << step << ' ' << i << ' ' << sample
                        << ' ' << sample_p.Value() << '\n';
                }
            }
        }

        double hit_ratio = static_cast<double>(likelihood_hits)
            / (likelihood_hits + likelihood_misses);
        std::cout << "\tLikelihood cache hit ratio : " << hit_ratio << "\n";

        if (step % rebuild_every == rebuild_every - 1) {
            parameters.clear();
            for (Index i : dp.get_parameters()) {
                parameters.push_back(*saved_fsts[i]);
            }
            saved_fsts[ngram] = new FST(ngram_counts(
                        parameters, ug_counts, alignemes, ug_weight));
            likelihood_cache.clear();
            compose_cache.clear();

            std::string fst_out_file = fst_out_dir + "fst_"
                + std::to_string(step) + ".fst";
            saved_fsts[ngram]->Write(fst_out_file);
        }
    }

    for (Index i = 0; i < static_cast<Index>(saved_fsts.size()); ++i) {
        Form form = fst_to_form(*saved_fsts[i]);
        ur_file << i << ' ' << display_form(form, phonemes);
    }
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("disclex");

    // TODO(padril): handle unimplemented geometric prior
    program.add_argument("--prior")
        .choices("geometric", "fst")
        .required()
        .help("which type of prior should be used");

    program.add_argument("--observations")
        .required()
        .help("the file containing the observed phonetic forms, each on its"
              "own line, with phones separated by spaces");

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

    // should these be FST only?
    program.add_argument("--universal-grammar-weight")
        .scan<'g', double>()
        .required()
        .help("how much to value the universal grammar relative to the "
              "learned phonology");
    program.add_argument("--rebuild-stride")
        .scan<'d', size_t>()
        .required()
        .help("how many steps to take between rebuilding the learned "
              "phonology");

    program.add_argument("--seed")
        .scan<'d', size_t>()
        .help("what seed to use for the RNG if any");

    program.add_argument("--output-deltas")
        .required()
        .help("the file to output DP process deltas to");
    program.add_argument("--output-parameters")
        .required()
        .help("the file to output saved parameters to");
    program.add_argument("--output-fsts-dir")
        .required()
        .help("the directory to output saved FSTs to");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::cout << "Successfully parsed args\n";

    Labelling<Label, Text, Text> phonemes(0, [](Label label){
            return ++label;
            });

    // TODO(padril): a better way to force epsilon to be 0 than just doing it
    //               first.
    std::string epsilon_ident = program.get<std::string>("--epsilon");
    phonemes.special("epsilon");
    if (epsilon_ident != "") {
        phonemes.associate_special("epsilon", epsilon_ident);
    } else {
        phonemes.associate_special("epsilon", "");
    }
    std::string phi_ident = program.get<std::string>("--phi");
    phonemes.special("phi");
    if (phi_ident != "") {
        phonemes.associate_special("phi", phi_ident);
    } else {
        phonemes.associate_special("phi", "");
    }
    // TODO(padril): add start and end implicitly if they're unset
    std::string start_ident = program.get<std::string>("--start");
    bool explicit_start;
    phonemes.special("start");
    if (start_ident != "") {
        explicit_start = true;
        phonemes.associate_special("start", start_ident);
    } else {
        explicit_start = false;
        phonemes.associate_special("start", "");
    }
    std::string end_ident = program.get<std::string>("--end");
    bool explicit_end;
    phonemes.special("end");
    if (end_ident != "") {
        explicit_end = true;
        phonemes.associate_special("end", end_ident);
    } else {
        explicit_end = false;
        phonemes.associate_special("end", "");
    }

    std::string output_deltas_path = program.get<std::string>(
            "--output-deltas");
    std::string output_parameters_path = program.get<std::string>(
            "--output-parameters");
    std::string output_fsts_dir = program.get<std::string>(
            "--output-fsts-dir");
    // TODO(padril): safely check that each arg is in a reasonable range
    // TODO(padril): these should probably be unicode capable
    std::string phonemes_path = program.get<std::string>("--phonemes");
    // TODO(padril): safely handle opening files
    IFStream phonemes_file(phonemes_path, std::ios_base::in);
    for (Text line; std::getline(phonemes_file, line);) {
        // TODO(padril): handle whitespace and blank lines
        phonemes.encode(line);
    }
    // TODO(padril): is there a way to defer this?
    phonemes_file.close();

    std::string phones_path = program.get<std::string>("--phones");
    IFStream phones_file(phonemes_path, std::ios_base::in);
    for (Text line; std::getline(phones_file, line);) {
        // TODO(padril): handle whitespace and blank lines
        phonemes.encode(line);
    }
    // TODO(padril): is there a way to defer this?
    phones_file.close();

    std::string observations_path = program.get<std::string>("--observations");

    size_t steps = program.get<size_t>("--steps");
    double alpha = program.get<double>("--alpha");
    double universal_grammar_weight = program.get<double>(
            "--universal-grammar-weight");
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

    IFStream observations_file(observations_path, std::ios_base::in);
    std::vector<Form> observations;
    for (Text line; std::getline(observations_file, line);) {
        // TODO(padril): This (and isomorphism) is not robust to end of line
        //               whitespace.
        std::vector<Text> segments = split(line, ' ');
        Form values;
        if (!explicit_start) {
            values.push_back(phonemes.special("start"));
        }
        for (Text segment : segments) {
            values.push_back(phonemes.encode(segment));
        }
        if (!explicit_end) {
            values.push_back(phonemes.special("end"));
        }
        observations.push_back(values);
    }
    // TODO(padril): is there a way to defer this?
    observations_file.close();

    // std::ofstream phones_out_file("phones_int.txt", std::ios_base::out);
    // for (Label label : phonemes.labels()) {
    //     phones_out_file << phonemes.decode(label) << ' ' << label << '\n';
    // }
    // phones_out_file.close();


    std::cout << "Successfully read surfaces file.\n";

    // TODO(padril): loading initial alignments from a file would be nice.
    // TODO(padril): a more sophisticated way of creating a universal grammar
    //               would be nice. an old example should be in past commits.

    Labelling<Label, std::pair<Label, Label>, Text> alignemes(
            0, [](Label label){ return ++label; });
    // TODO(padril): Move to native FST Symbol maps
    {
        alignemes.special("phi");
        auto phi = phonemes.special("phi");
        alignemes.associate_special("phi", std::pair(phi, phi));
        alignemes.special("epsilon");
        auto epsilon = phonemes.special("epsilon");
        alignemes.associate_special("epsilon", std::pair(epsilon, epsilon));
    }
    NGramFST ug;
    {
        FST top;
        for (Label phoneme : phonemes.labels()) {
            if (phoneme == phonemes.special("phi") ||
                    phoneme == phonemes.special("start") ||
                    phoneme == phonemes.special("end")) {
                continue;
            }
            for (Label phone : phonemes.labels()) {
                if (phone == phonemes.special("phi") ||
                        phone == phonemes.special("start") ||
                        phone == phonemes.special("end")) {
                    continue;
                }
                if (phoneme == phonemes.special("epsilon") &&
                        phone == phonemes.special("epsilon")) {
                    continue;
                }
                Form ur = Form({
                        phonemes.special("start"),
                        phoneme,
                        phonemes.special("end")});
                Form sr = Form({
                        phonemes.special("start"),
                        phone,
                        phonemes.special("end")});
                fst::Union(&top, to_acceptor(alignemes, align_to_fst(ur, sr)));
            }
        }
        fst::RmEpsilon(&top);
        fst::Determinize(top, &top);

        ngram::NGramCounter<NGramFST::Arc::Weight, Label> counter(1);
        if (!counter.Count(top)) {
            std::cerr << "Count n-gram FST (UG) could not be properly "
                         "computed\n";
        }

        counter.GetFst(&ug);

        Label backoff_label = alignemes.special("phi");
        fst::ArcSort(&ug, fst::ILabelCompare<NGramFST::Arc>());
        ngram::NGramKneserNey make(&ug, false, backoff_label);
        // TODO(padril): look at the other options of this
        // TODO(padril): maybe use ngram shrink
        make.MakeNGramModel();
    }

    std::cout << "Successfully constructed universal grammar.\n";
    std::cout << "Starting Gibbs sampling:\n";

    Lexicon lexicon;
    std::ofstream dp_file(output_deltas_path, std::ios_base::out);
    std::ofstream ur_file(output_parameters_path, std::ios_base::out);
    gibbs_fst_dijkstra(
            steps, engine, observations, ug, alignemes,
            phonemes, alpha, universal_grammar_weight, rebuild_stride,
            dp_file, ur_file, output_fsts_dir);

    std::cout << "Completed Gibbs sampling.\n";

    // std::cout << ctrack::result_as_string();
}

