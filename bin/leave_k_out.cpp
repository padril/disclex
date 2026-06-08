// TODO(padril): a lot of things in this file can be shared with disclex
#include <string>
#include <fstream>

#include "labelling.hpp"

#include "external/argparse.hpp"
#include <fst/fstlib.h>
#include <ngram/ngram.h>

using Label = int;
using Char = char;
using Text = std::string;
using IFStream = std::basic_ifstream<Char>;
using Semiring = fst::LogWeightTpl<float>;
using Arc = fst::ArcTpl<Semiring, Label>;
template <typename A>
using FSTType = fst::VectorFst<A>;
using FST = FSTType<Arc>;
using NGramFST = fst::StdVectorFst;
using Form = std::vector<Label>;
using SR = FST;
using UR = FST;
using AR = FST;

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

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("leave_k_out");

    program.add_argument("--lexicon")
        .required()
        .help("the file containing the observed phonetic forms and the gold"
              "phonemic forms separated by tabs, each on its own line, with "
              "phones/phonemes separated by spaces");

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

    // should these be FST only?
    program.add_argument("--universal-grammar-weight")
        .scan<'g', double>()
        .required()
        .help("how much to value the universal grammar relative to the "
              "learned phonology");

    program.add_argument("--output")
        .required()
        .help("the file to output leave-k-out datapoints to");

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

    std::string output_path = program.get<std::string>(
            "--output");

    double ug_weight = program.get<double>(
            "--universal-grammar-weight");

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
    IFStream phones_file(phones_path, std::ios_base::in);
    for (Text line; std::getline(phones_file, line);) {
        // TODO(padril): handle whitespace and blank lines
        phonemes.encode(line);
    }
    // TODO(padril): is there a way to defer this?
    phones_file.close();

    std::string lexicon_path = program.get<std::string>("--lexicon");
    IFStream lexicon_file(lexicon_path, std::ios_base::in);
    std::vector<std::pair<Form, Form>> lexicon;
    for (Text line; std::getline(lexicon_file, line);) {
        // TODO(padril): This (and isomorphism) is not robust to end of line
        //               whitespace.
        std::vector<Text> sr_ur = split(line, '\t');
        Text sr = sr_ur[0];
        Text ur = sr_ur[1];
        std::vector<Text> sr_segments = split(sr, ' ');
        std::vector<Text> ur_segments = split(ur, ' ');
        Form sr_values;
        Form ur_values;
        if (!explicit_start) {
            sr_values.push_back(phonemes.special("start"));
            ur_values.push_back(phonemes.special("start"));
        }
        for (Text segment : sr_segments) {
            sr_values.push_back(phonemes.encode(segment));
        }
        for (Text segment : ur_segments) {
            ur_values.push_back(phonemes.encode(segment));
        }
        if (!explicit_end) {
            sr_values.push_back(phonemes.special("end"));
            ur_values.push_back(phonemes.special("end"));
        }
        lexicon.push_back(std::pair(sr_values, ur_values));
    }
    // TODO(padril): is there a way to defer this?
    lexicon_file.close();

    std::cout << "Successfully read lexicon file.\n";

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

    std::ofstream output(output_path, std::ios_base::out);
    output << "k i hyp ref distance\n";
    for (size_t k = 0; k < lexicon.size(); ++k) {
        std::cout << "Starting k=" << k << " ";
        std::vector<std::pair<Form, Form>> left_out_forms(lexicon.begin(), lexicon.begin() + k);
        std::vector<std::pair<Form, Form>> left_in_forms(lexicon.begin() + k, lexicon.end());
        std::vector<UR> train_urs;
        for (auto [_, ur_form] : left_in_forms) {
            train_urs.push_back(form_to_fst(ur_form));
        }
        FST trained_fst = ngram_counts(train_urs, ug, alignemes, ug_weight);
        std::cout << ".";
        for (size_t i = 0; i < k; ++i) {
            if (i == k / 3 || i == 2 * k / 3) { std::cout << "."; }
            UR sr = form_to_fst(left_out_forms[i].first);
            FST fst_sr;
            fst::Compose(trained_fst, sr, &fst_sr);
            FST predicted_ur;

            fst::StdVectorFst predicted_ur_trop;
            fst::StdVectorFst fst_sr_trop;
            fst_sr_trop.SetStart(fst_sr.Start());
            for (fst::StateIterator<FST> siter(fst_sr); !siter.Done(); siter.Next()) {
                auto s = siter.Value();
                fst_sr_trop.AddState();
                if (fst_sr.Final(s) != Semiring::Zero()) {
                    float log_final = fst_sr.Final(s).Value();
                    fst_sr_trop.SetFinal(s, fst::TropicalWeight(std::max(0.0f, log_final)));
                }
                for (fst::ArcIterator<FST> aiter(fst_sr, s); !aiter.Done(); aiter.Next()) {
                    const auto &arc = aiter.Value();
                    float log_w = arc.weight.Value();
                    float cost = std::max(0.0f, log_w);
                    fst::StdArc new_arc(
                        arc.ilabel,
                        arc.olabel,
                        fst::TropicalWeight(cost),
                        arc.nextstate
                    );
                    fst_sr_trop.AddArc(s, new_arc);
                }
            }

            fst::ShortestPath(fst_sr_trop, &predicted_ur_trop);

            fst::ArcMap(predicted_ur_trop, &predicted_ur, fst::StdToLogMapper());

            Form predicted_ur_form = fst_to_form(predicted_ur);
            predicted_ur_form.erase(
                    std::remove_if(
                        predicted_ur_form.begin(),
                        predicted_ur_form.end(),
                        [&phonemes](Label x) -> bool {
                            return x == phonemes.special("epsilon");
                        }),
                    predicted_ur_form.end());
            Form gold_ur_form = left_out_forms[i].second;
            gold_ur_form.erase(
                    std::remove_if(
                        gold_ur_form.begin(),
                        gold_ur_form.end(),
                        [&phonemes](Label x) -> bool {
                            return x == phonemes.special("epsilon");
                        }),
                    gold_ur_form.end());
            int distance = levenshtein(predicted_ur_form, gold_ur_form);
            output << k << ' '
                << i << ' '
                << display_form(predicted_ur_form, phonemes, "", "", "") << ' '
                << display_form(gold_ur_form, phonemes, "", "", "") << ' '
                << distance << '\n' << std::flush;
        }
        std::cout << " Done!\n";
    }
    output.close();
    
    std::cout << "Done.\n";
}
