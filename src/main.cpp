#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <functional>
#include "distribution.hpp"
#include <fst/fstlib.h>
#include <ngram/ngram.h>
#include "ctrack.hpp"
#include "labelling.hpp"
#include "argparse.hpp"

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

// // TODO: replace these segment types with maps probably
// class Phon : public segment::TextMappedSegment<Label, Text>,
//     public distribution::Distribution<Label, Semiring, Engine, std::monostate> {
// private:
//     distribution::Distribution<Semiring, std::monostate, Engine, std::monostate>* uniform;
// public:
//     Label epsilon;
//     Label phi;
//     Label start;
//     Label end;
//     // TODO: this pattern of a categorical distribution is common, maybe abstract it
//     Phon(IFStream& file, distribution::Distribution<Semiring, std::monostate, Engine, std::monostate>* uniform,
//             Label epsilon, Label phi, Label start, Label end)
//         : uniform(uniform), epsilon(epsilon), phi(phi), start(start), end(end)
//     {
//         using Pair = std::tuple<Label, Text>;
//         // TODO: this imbue is probably not optimal. what if they don't pass in utf-8?
//         file.imbue(std::locale());
//         std::vector<Pair> pairs = parse(file, [](Text) { return 0; });
//         _texts.resize(pairs.size());
//         std::transform(pairs.begin(), pairs.end(), _texts.begin(),
//                 [](Pair p) { return std::get<Text>(p); });
//     };
//     size_t size() override { return _texts.size(); }
//     Form values() override {
//         Form ret(_texts.size());
//         std::iota(ret.begin(), ret.end(), 0);
//         return ret;
//     }
//     Text isomorphism(Label value) override {
//         return _texts[value];
//     }
//     Label isomorphism(Text value) override {
//         return std::find(_texts.begin(), _texts.end(), value) - _texts.begin();
//     }
//     std::pair<Label, Semiring> sample(Engine& engine, std::monostate) override {
//         Semiring next = std::get<Semiring>(uniform->sample(engine, std::monostate()));
//         Semiring bound = Semiring::Zero();
//         for (Label value : values()) {
//             // TODO: better way to filter
//             if (value == epsilon || value == phi || value == start || value == end) { continue; }
//             bound += Semiring(-std::log(1.f / (size() - 4)));
//             if (next <= bound) {
//                 return std::pair(value, bound);
//             }
//         }
//         // unreachable
//         return std::pair(values().back(), 0);
//     }
// private:
//     std::vector<Text> _texts;
// };

class UniformSemiring : public distribution::Distribution<Semiring, std::monostate, Engine, std::monostate> {
private:
    std::uniform_real_distribution<Semiring::ValueType> uniform{0.0, 1.0};
public:
    UniformSemiring() { } ;
    std::pair<Semiring, std::monostate> sample(Engine& engine, std::monostate) override {
        return std::pair(Semiring(-std::log(uniform(engine))), std::monostate());
    }
};

// TODO: For geom, can be moved
// // TODO: const reference stuff everywhere
// double get_nlld(Phon &phones, Form &ur_, Form &sr_, MatrixXd &transform) {
//     // transform  :  phones.size() x phonemes.size()
//     MatrixXd ur = ur_.onehot();  // phonemes.size() x repr.MAX_LENGTH
//     MatrixXd observed = sr_.onehot(); // phones.size() x repr.MAX_LENGTH
//     MatrixXd expected(phones.size(), repr::MAX_LENGTH);
//     MatrixXd probs(repr::MAX_LENGTH, repr::MAX_LENGTH);
// 
//     expected.noalias() = transform * ur;
//     probs.noalias() = expected.transpose() * observed;
// 
//     double nlld = 0;
//     // TODO: another static cast here that might need fixing
//     for (ulong j = 0; j < repr::MAX_LENGTH; ++j) {
//         nlld -= std::log10(probs(j, j) + EPS);
//     }
//     return nlld;
// }

// TODO: std library or boost might have something for this?
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
};

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
};

Form fst_to_form(FST fst) {
    Form form;

    // fst::StdVectorFst trop_fst;
    // fst::StdVectorFst best;
    // fst::ArcMap(log_fst, &trop_fst, fst::LogToStdMapper());
    // fst::ShortestPath(trop_fst, &best);

    auto state = fst.Start();
    if (state == fst::kNoStateId) return form;

    while (true) {
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


void display_form(
        Form form, Labelling<Label, Text, Text>& labelling,
        std::string lwrap="", std::string rwrap="", std::string end="\n")
{
    std::cout << lwrap;
    for (Label segment : form) {
        if (segment == labelling.special("epsilon")) { continue; }
        std::cout << labelling.decode(segment);
    }
    std::cout << rwrap << end;
}

template <typename A>
fst::VectorFst<A> to_acceptor(
        Labelling<Label, std::pair<Label, Label>, Text>& labelling,
        fst::VectorFst<A> fst)
{
    fst::VectorFst<A> result;

    for (fst::StateIterator<fst::VectorFst<A>> siter(fst); !siter.Done(); siter.Next()) {
        result.AddState();
    }

    result.SetStart(fst.Start());

    for (fst::StateIterator<fst::VectorFst<A>> siter(fst); !siter.Done(); siter.Next()) {
        auto s = siter.Value();
        if (fst.Final(s) != A::Weight::Zero()) {
            result.SetFinal(s, A::Weight::One()); // keep weight 0, adjust if needed
        }

        for (fst::ArcIterator<fst::VectorFst<A>> aiter(fst, s); !aiter.Done(); aiter.Next()) {
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
        fst::VectorFst<A> fst)
{
    fst::VectorFst<A> result;

    for (fst::StateIterator<fst::VectorFst<A>> siter(fst); !siter.Done(); siter.Next()) {
        result.AddState();
    }

    result.SetStart(fst.Start());

    for (fst::StateIterator<fst::VectorFst<A>> siter(fst); !siter.Done(); siter.Next()) {
        auto s = siter.Value();
        if (fst.Final(s) != A::Weight::Zero()) {
            result.SetFinal(s, A::Weight::One());
        }

        for (fst::ArcIterator<fst::VectorFst<A>> aiter(fst, s); !aiter.Done(); aiter.Next()) {
            auto a = aiter.Value();
            auto pair = labelling.decode(a.ilabel);
            result.AddArc(s, A(pair.first, pair.second, a.weight, a.nextstate));
        }
    }

    return result;
}

// TODO: pass in phon or whatever we end up calling it for phi and epsilon
// TODO: Universal grammar should be a simple FST
FST ngram_counts(
        Lexicon lexicon, NGramFST ug_model,
        Labelling<Label, std::pair<Label, Label>, Text>& labelling,
        double ug_weight)
{
    FST top;

    for (auto [ur, _] : lexicon) {
        fst::Union(&top, to_acceptor(labelling, ur));
    }
    fst::RmEpsilon(&top);
    fst::Determinize(top, &top);

    // TODO: generalize order
    // TODO: look into .AddSequence
    ngram::NGramCounter<NGramFST::Arc::Weight, Label> counter(2);
    if (!counter.Count(top)) {
        std::cerr << "Count n-gram FST could not be properly computed\n";
    };

    NGramFST model;
    counter.GetFst(&model);

    // Must be one of:
    // - kneser_ney
    // - absolute
    // - katz
    // - witten_bell
    // - unsmoothed
    // - presmoothed
    // TODO: allow other smoothing methods
    std::string method = "kneser_ney";

    Label backoff_label = labelling.special("phi");
    fst::ArcSort(&model, fst::ILabelCompare<NGramFST::Arc>());
    ngram::NGramKneserNey make(&model, false, backoff_label);
    // TODO: look at the other options of this
    // TODO: maybe use ngram shrink
    make.MakeNGramModel();

    ngram::NGramModelMerge merger(&model, backoff_label);
    // TODO: check if alpha is in the right order
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
    // TODO: Implement the 'Exact Offline Representation' as an option using
    // ngram convert
    for (fst::StateIterator<FST> siter(transform); !siter.Done(); siter.Next()) {
        FST::StateId state = siter.Value();
        for (fst::MutableArcIterator<FST> aiter(&transform, state); !aiter.Done(); aiter.Next()) {
            Arc arc = aiter.Value();
            if (arc.ilabel == phi) { arc.ilabel = epsilon; }
            if (arc.olabel == phi) { arc.olabel = epsilon; }
            aiter.SetValue(arc);
        }
    }

    // TODO: probabilities are not actually normalized right now because of
    // epsilon issues (or some other rounding accumulation) I think

    return transform;
}

std::pair<FST, Semiring> random_walk(Engine& engine, FST fst) {
    // TODO: might want to normalize using forward-backward first
    UniformSemiring uniform;

    UR ur;
    Semiring p = Semiring::One();
    FST::StateId ur_curr = ur.AddState();
    ur.SetStart(ur_curr);

    // TODO: fix this being needed to prevent the bug
    fst::RmEpsilon(&fst);

    // fst::RandGen does not return probability weights, so we need to write
    // this ourselves
    FST::StateId curr = fst.Start();

    // TODO: there is a sampling bug where we get a segfault when we hit max length
    const size_t  MAX_LENGTH    = 150;

    for (size_t n = 0; n < MAX_LENGTH; ++n) {
        // TODO: this is inefficient, could we prenormalize somewhere? idk
        Semiring final = fst.Final(curr);
        Semiring sum = final;
        for (fst::ArcIterator<FST> aiter(fst, curr); !aiter.Done(); aiter.Next()) {
            Arc arc = aiter.Value();
            sum += arc.weight;
        }
        Semiring bound = final;
        Semiring next = std::get<Semiring>(uniform.sample(engine, std::monostate())) * sum;
        if (next <= bound) {  // Final probability got hit
            p *= bound;
            ur.SetFinal(ur_curr, Semiring::One());
            fst::ArcSort(&ur, fst::ILabelCompare<Arc>());
            return std::pair(ur, p);
        }
        // Assume that the outgoing arcs are correctly set
        for (fst::ArcIterator<FST> aiter(fst, curr); !aiter.Done(); aiter.Next()) {
            Arc arc = aiter.Value();
            bound += arc.weight;
            if (next <= bound) {
                FST::StateId ur_next = ur.AddState();
                // TODO: idk if this should actually be Semiring::One()
                ur.AddArc(ur_curr, Arc(arc.ilabel, arc.ilabel, Semiring::One(), ur_next));
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
    // CTRACK;
    FST ret;
    fst::Compose(left, right, &ret);

    return ret;
};

Semiring likelihood(FST fst, SR sr, UR ur) {
    FST composed = compose(ur, compose(fst, sr));
    std::vector<Semiring> dists;
    fst::ShortestDistance(composed, &dists, true);
    return dists[composed.Start()];
}

// TODO: could be way better, maybe using boost
struct HashPair {
    template <class T, class U>
    std::size_t operator()(const std::pair<T, U> &p) const {
        auto h1 = std::hash<T>{}(p.first);
        auto h2 = std::hash<U>{}(p.second);
        return h1 ^ h2;  
    }
};


// TODO: split Form into UR and SR forms?
// and maybe better name SR and UR
FST gibbs_fst_dijkstra(
        Lexicon* lexicon,
        int steps, double burn_in, Engine engine, std::vector<Form> surface_forms,
        NGramFST ug_counts,
        Labelling<Label, std::pair<Label, Label>, Text>& alignemes,
        Semiring alpha,
        double ug_weight,
        size_t rebuild_every
        )
{
    using Index = int;

    std::vector<FST> saved_fsts;

    saved_fsts.push_back(FST());
    Index ngram = saved_fsts.size() - 1;

    // TODO: Lexicon should be its own abstraction
    lexicon->clear();
    std::vector<Index> observations;
    std::vector<Index> parameters;
    for (Form sr : surface_forms) {
        UR ur_fst = form_to_fst(sr);
        SR sr_fst = form_to_fst(sr);
        fst::ArcSort(&ur_fst, fst::OLabelCompare<Arc>());
        fst::ArcSort(&sr_fst, fst::ILabelCompare<Arc>());
        lexicon->push_back(std::pair(ur_fst, Semiring::Zero()));
        saved_fsts.push_back(ur_fst);
        parameters.push_back(saved_fsts.size() - 1);
        saved_fsts.push_back(sr_fst);
        observations.push_back(saved_fsts.size() - 1);
    }
    std::cout << "\tCompleted initial alignment.\n";

    UniformSemiring uniform;

    std::unordered_map<std::pair<Index, Index>, Semiring, HashPair> likelihood_cache;
    std::unordered_map<std::pair<Index, Index>, Index, HashPair> compose_cache;

    distribution::MapGivenDistribution< Index, Semiring, Engine, Index, std::pair<std::optional<Index>, std::optional<Index>>> prior(
        new distribution::FSTDistribution<Index, Semiring, Engine, Index>(
            &ngram,
            [&saved_fsts](Engine& engine, Index fst) -> std::pair<Index, Semiring> {
                UR ur;
                Semiring p;
                std::tie(ur, p) = random_walk(engine, saved_fsts[fst]);
                saved_fsts.push_back(ur);
                return std::pair(saved_fsts.size() - 1, p);
            },
            [&saved_fsts, &compose_cache](Index left, Index right) -> Index {
                if (auto it = compose_cache.find(std::pair(left, right)); it != compose_cache.end()) {
                    return it->second;
                }
                FST composed_fst = compose(saved_fsts[left], saved_fsts[right]);
                saved_fsts.push_back(composed_fst);
                Index composed = saved_fsts.size() - 1;
                compose_cache[std::pair(left, right)] = composed;
                return composed;
            }
            ),
        [](Index sr) {
            return std::pair(std::nullopt, sr);
        }
        );

    distribution::GibbsDirichletProcess<Index, Semiring, Engine, Index, Index, LexiconMap> dp(
            &prior,
            alpha,
            observations,
            parameters,
            [&saved_fsts, &likelihood_cache, &ngram](Index sr, Index ur) -> Semiring {
                if (auto it = likelihood_cache.find(std::pair(sr, ur)); it != likelihood_cache.end()) {
                    return it->second;
                }
                Semiring l = likelihood(saved_fsts[ngram], saved_fsts[sr], saved_fsts[ur]);
                likelihood_cache[std::pair(sr, ur)] = l;
                return l;
            },
            [&saved_fsts, &compose_cache, &ngram](Index sr) -> Semiring {
                // TODO: cache this as well
                FST composed_fst;
                if (auto it = compose_cache.find(std::pair(ngram, sr)); it != compose_cache.end()) {
                    composed_fst = saved_fsts[it->second];
                } else {
                    composed_fst = compose(saved_fsts[ngram], saved_fsts[sr]);
                    saved_fsts.push_back(composed_fst);
                    Index composed = saved_fsts.size() - 1;
                    compose_cache[std::pair(ngram, sr)] = composed;
                }
                std::vector<Semiring> dists;
                // TODO: make shortestdistance gracefully exit when no dists found
                fst::ShortestDistance(composed_fst, &dists, true);
                return dists[composed_fst.Start()];
            },
            &uniform
            );

    saved_fsts[ngram] = ngram_counts(*lexicon, ug_counts, alignemes, ug_weight);
    std::cout << "\tCompleted initial NGram count.\n";

    for (int step = 0; step < steps; ++step) {
        std::cout << "\tStarting step " << step << ".\n";
        // TODO: better cast here
        for (size_t i = 0; i < observations.size(); ++i) {
            Index sample;
            Semiring sample_p;
            std::tie(sample, sample_p) = dp.sample(engine, i);
            dp.update(i, sample);

            std::pair<UR, Semiring> best = (*lexicon)[i];
            if (step < steps * burn_in) {
                if (sample_p > std::get<Semiring>(best)) {
                    (*lexicon)[i] = std::pair(saved_fsts[sample], sample_p);
                }
            }
        }
        if (step % rebuild_every == rebuild_every - 1) {
            saved_fsts[ngram] = ngram_counts(*lexicon, ug_counts, alignemes, ug_weight);
            likelihood_cache.clear();
            compose_cache.clear();
        }
    }

    return saved_fsts[ngram];
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("disclex");

    // TODO: handle unimplemented geometric prior
    program.add_argument("--prior")
        .choices("geometric", "fst")
        .required()
        .help("which type of prior should be used");

    // TODO: a script to automatically generate this from a directory containing text files
    program.add_argument("--observations")
        .required()
        .help("the file containing the observed phonetic forms, each on its"
              "own line, with phones separated by spaces");

    // TODO: maybe infer this from the input wordlist
    program.add_argument("--phones")
        .required()
        .help("the file containing the list of phones, each on its own line");

    program.add_argument("--phonemes")
        .required()
        .help("the file containing the list of phonemes, each on its own line");

    // TODO: handle reading these from the phones file
    program.add_argument("--epsilon")
        .default_value("")
        .help("the string identifying epsilon in phone(me) files, if made explicit");
    program.add_argument("--phi")
        .default_value("")
        .help("the string identifying phi in phone(me) files, if made explicit");
    program.add_argument("--start")
        .default_value("")
        .help("the string identifying the start token in phone(me) files, if made explicit");
    program.add_argument("--end")
        .default_value("")
        .help("the string identifying the end token in phone(me) files, if made explicit");

    program.add_argument("--geometric-stop-chance")
        .scan<'g', double>()
        .default_value(0.2)
        .help("the stop chance for the geometric prior, if being used");
    // TODO: add to help comment about infinite looping and possibility of using -1 or --geometric-no-max-length
    program.add_argument("--geometric-max-length")
        .scan<'d', size_t>()
        .default_value(16)
        .help("the maximum length for the geometric prior, if being used");

    program.add_argument("--steps")
        .scan<'d', size_t>()
        .required()
        .help("the number of gibbs steps to do");
    program.add_argument("--burn-in")
        .scan<'g', double>()
        .required()
        .help("the percent of the steps to be used as burn in, given as a value between 0.0 and 1.0");
    program.add_argument("--alpha")
        .scan<'g', double>()
        .required()
        .help("the alpha parameter for the dirichlet process");

    // should these be FST only?
    program.add_argument("--universal-grammar-weight")
        .scan<'g', double>()
        .required()
        .help("how much to value the universal grammar relative to the learned phonology");
    program.add_argument("--rebuild-stride")
        .scan<'d', size_t>()
        .required()
        .help("how many steps to take between rebuilding the learned phonology");

    program.add_argument("--seed")
        .scan<'d', size_t>()
        .help("what seed to use for the RNG if any");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::cout << "Successfully parsed args\n";

    Labelling<Label, Text, Text> phonemes(0, [](Label label){ return ++label; });

    // TODO: a better way to force epsilon to be 0 than just doing it first
    std::string epsilon_ident = program.get<std::string>("--epsilon");
    phonemes.special("epsilon");
    if (epsilon_ident != "") {
        // TODO: associating with an already present label should change the
        // label of the special
        phonemes.associate_special("epsilon", epsilon_ident);
    }
    std::string phi_ident = program.get<std::string>("--phi");
    phonemes.special("phi");
    if (phi_ident != "") {
        phonemes.associate_special("phi", phi_ident);
    }
    // TODO: add start and end implicitly if they're unset
    std::string start_ident = program.get<std::string>("--start");
    phonemes.special("start");
    if (start_ident != "") {
        phonemes.associate_special("start", start_ident);
    }
    std::string end_ident = program.get<std::string>("--end");
    phonemes.special("end");
    if (end_ident != "") {
        phonemes.associate_special("end", end_ident);
    }

    // TODO: safely check that each arg is in a reasonable range
    // TODO: these should probably be unicode capable
    std::string phonemes_path = program.get<std::string>("--phonemes");
    // TODO: safely handle opening files
    IFStream phonemes_file(phonemes_path, std::ios_base::in);
    for (Text line; std::getline(phonemes_file, line);) {
        // TODO: handle whitespace and blank lines
        phonemes.encode(line);
    }
    // TODO: is there a way to defer this?
    phonemes_file.close();

    std::string phones_path = program.get<std::string>("--phones");
    IFStream phones_file(phonemes_path, std::ios_base::in);
    for (Text line; std::getline(phones_file, line);) {
        // TODO: handle whitespace and blank lines
        phonemes.encode(line);
    }
    // TODO: is there a way to defer this?
    phones_file.close();

    std::string observations_path = program.get<std::string>("--observations");

    size_t steps = program.get<size_t>("--steps");
    double burn_in = program.get<double>("--burn-in");
    double alpha = program.get<double>("--alpha");
    double universal_grammar_weight = program.get<double>("--universal-grammar-weight");
    size_t rebuild_stride = program.get<size_t>("--rebuild-stride");

    // TODO: I don't know if this works
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
        // TODO: This (and isomorphism) is not robust to end of line whitespace
        std::vector<Text> segments = split(line, ' ');
        Form values;
        for (Text segment : segments) {
            values.push_back(phonemes.encode(segment));
        }
        observations.push_back(values);
    }
    // TODO: is there a way to defer this?
    observations_file.close();

    std::ofstream phones_out_file("phones_int.txt", std::ios_base::out);
    for (Label label : phonemes.labels()) {
        phones_out_file << phonemes.decode(label) << ' ' << label << '\n';
    }
    phones_out_file.close();


    std::cout << "Successfully read surfaces file.\n";

    // TODO: get initial alignments in a better way
    // std::vector<segment::AlignmentString<Label, Label, std::wstring>> alignments(surfaces.size());
    // std::transform(surfaces.begin(), surfaces.end(), alignments.begin(), [](Form form) {
    //             return segment::identity(form.segments, "|");
    //         });
    
    // TODO: lots of const references that need to be thrown around everywhere
    // TODO: keep geom stuff as a flag given in main
    // distribution::Geometric<Label, Random> geom(rand_engine, STOP_CHANCE, MAX_LENGTH, phonemes);

    // FST universal_grammar_log;
    // {
    //     auto main = universal_grammar_log.AddState();
    //     auto start = universal_grammar_log.AddState();
    //     universal_grammar_log.SetStart(start);
    //     universal_grammar_log.AddArc(start, Arc(PHI, PHI, Semiring::One(), main));

    //     for (Label phoneme : phonemes.values()) {
    //         auto state = universal_grammar_log.AddState();
    //         universal_grammar_log.AddArc(main, Arc(phoneme, NOTHING, UNIVERSAL_GRAMMAR_WEIGHT, state));
    //         universal_grammar_log.AddArc(state, Arc(PHI, PHI, Semiring::One(), main));
    //     }
    //     for (Label phone : phones.values()) {
    //         auto state = universal_grammar_log.AddState();
    //         universal_grammar_log.AddArc(main, Arc(NOTHING, phone, UNIVERSAL_GRAMMAR_WEIGHT, state));
    //         universal_grammar_log.AddArc(state, Arc(PHI, PHI, Semiring::One(), main));
    //     }

    //     universal_grammar_log.SetFinal(start, Semiring::Zero());
    //     // TODO: Some normalization step should almost certainly occur here
    // }
    // fst::StdVectorFst universal_grammar;
    // std::vector<std::pair<Label, Label>> table;
    // fst::ArcMap(
    //     to_acceptor(universal_grammar_log, table),
    //     &universal_grammar,
    //     fst::SemiringConvertMapper<Arc, fst::StdArc>()
    // );
    // fst::Determinize(universal_grammar, &universal_grammar);
   
    Labelling<Label, std::pair<Label, Label>, Text> alignemes(
            0, [](Label label){ return ++label; });
    // TODO: Move to native FST Symbol maps
    alignemes.special("phi");
    alignemes.associate_special("phi", std::pair(phonemes.special("phi"), phonemes.special("phi")));
    alignemes.special("epsilon");
    alignemes.associate_special("epsilon", std::pair(phonemes.special("epsilon"), phonemes.special("epsilon")));
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
                Form ur = Form({phonemes.special("start"), phoneme, phonemes.special("end")});
                Form sr = Form({phonemes.special("start"), phone, phonemes.special("end")});
                fst::Union(&top, to_acceptor(alignemes, align_to_fst(ur, sr)));
            }
        }
        fst::RmEpsilon(&top);
        fst::Determinize(top, &top);

        ngram::NGramCounter<NGramFST::Arc::Weight, Label> counter(1);
        if (!counter.Count(top)) {
            std::cerr << "Count n-gram FST (UG) could not be properly computed\n";
        };

        counter.GetFst(&ug);

        Label backoff_label = alignemes.special("phi");
        fst::ArcSort(&ug, fst::ILabelCompare<NGramFST::Arc>());
        ngram::NGramKneserNey make(&ug, false, backoff_label);
        // TODO: look at the other options of this
        // TODO: maybe use ngram shrink
        make.MakeNGramModel();
    }

    std::cout << "Successfully constructed universal grammar.\n";
    std::cout << "Starting Gibbs sampling:\n";

    Lexicon lexicon;
    FST fst = gibbs_fst_dijkstra(
            &lexicon, steps, burn_in, engine, observations, ug, alignemes, alpha,
            universal_grammar_weight, rebuild_stride);

    std::cout << "Completed Gibbs sampling.\n";
    std::cout << "Displaying:\n";


    // Display first and last assignments
    for (ulong form = 0; form < observations.size(); ++form) {
        Form sr = observations[form];
        FST urfst;
        Semiring likelihood;
        std::tie(urfst, likelihood) = lexicon[form];
        std::vector<Semiring> dist;
        Form ur = fst_to_form(urfst);
        display_form(sr, phonemes, "[", "]", "\t");
        display_form(ur, phonemes, "\t/", "/", "\t");
        std::cout << likelihood.Value() << "\n";
    }

    // TODO: consider switching from wide chars to u8
    std::cout << ctrack::result_as_string();
}

