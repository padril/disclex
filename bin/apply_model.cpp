#include "impl/segment.hpp"
#include "labelling.hpp"

#include <ctime>
#include "impl/read_model.hpp"
#include "impl/make_model.hpp"
#include "external/argparse.hpp"

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("disclex");

    program.add_argument("--observations")
        .required()
        .help("the file containing the observations. see format "
              "specification in the README");
    program.add_argument("--model")
        .required()
        .append()
        .help("a file containing alignment forms. see format "
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

    bool sample = false;
    program.add_argument("--sample")
        .store_into(sample)
        .required()
        .help("should lexicon be constructed by sampling instead of by "
              "taking the best path");

    program.add_argument("--step")
        .scan<'d', size_t>()
        .help("what step the model was built on (used for mix portions)");
    program.add_argument("--seed")
        .scan<'d', size_t>()
        .help("what seed to use for the RNG if any");

    program.add_argument("--output-lexicon")
        .required()
        .help("the file to output the lexicon to");

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

    std::string output_lexicon_path = program.get<std::string>(
            "--output-lexicon");

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

    std::string observations_path = program.get<std::string>("--observations");
    std::vector<std::string> model_paths = program.get<std::vector<std::string>>("--model");
    std::string splits_path = program.get<std::string>("--splits");

    size_t step = program.get<size_t>("--step");

    // TODO(padril): I don't know if this works
    std::default_random_engine engine;
    if (program.is_used("--seed")) {
        engine.seed(program.get<size_t>("--seed"));
    } else {
        // TODO(padril): add this to disclex too
        engine.seed(std::time(nullptr));
    }

    std::vector<Alignment> alignments = read_model(
            model_paths,
            phonemes,
            phones
            );
    std::cout << "Successfully read model file.\n";

    std::unordered_map<std::string, Split> splits = read_splits(splits_path);
    std::cout << "Successfully read splits file.\n";

    size_t n = alignments.size();
    std::vector<std::tuple<FST<Log, Phoneme>, FST<Log, Phone>,
        std::string>> assignments(n);
    for (size_t i = 0; i < n; ++i) {
        auto [alignment, split] = alignments[i];
        auto [parameter_labels, observation_labels] = unzip(alignment);
        auto parameter = labels_to_path_fst<Log>(parameter_labels);
        auto observation = labels_to_path_fst<Log>(observation_labels);

        assignments[i] = std::tuple(parameter, observation, split);
    }

    // TODO(padril): pull the creation of an edit_fst out to another function
    FST<Log, Segment> edit_fst;  // The edit distance FST
    {
        auto s = edit_fst.AddState();
        edit_fst.SetStart(s);
        edit_fst.SetFinal(s, Log::One());

        for (Phoneme phoneme : phonemes.labels()) {
            for (Phone phone : phones.labels()) {
                if (phoneme == phonemes.special("epsilon") &&
                        phone == phones.special("epsilon")) { continue; }
                edit_fst.AddArc(s, decltype(edit_fst)::Arc(phoneme, phone, Log::One(), s));
            }
        }

        fst::ArcSort(&edit_fst, fst::OLabelCompare<decltype(edit_fst)::Arc>());
    }

    // TODO(padril): move alignemes out to another file too
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

    FST<Log, Segment> model = count_ngrams(
            assignments,
            splits,
            alignemes,
            edit_fst,
            step
            );
    std::cout << "Successfully built model.\n";

    std::ifstream observations_file(observations_path, std::ios_base::in);
    std::ofstream lexicon_file(output_lexicon_path, std::ios_base::out);
    for (std::string observation_string; getline(observations_file, observation_string); ) {
        std::vector<Phone> observation_labels;
        std::istringstream observation_stream(observation_string);
        for (std::string segment; getline(observation_stream, segment, ' ');) {
            observation_labels.push_back(phones.encode(segment));
        }

        auto observation = labels_to_path_fst<Log>(observation_labels);
        auto conditioned = compose(model, observation);
        FST<Log, Segment> path;
        if (sample) {
            path = random_walk(engine, conditioned).first;
        } else {
            FST<Trop, Segment> trop_path;
            fst::ShortestPath(log_to_trop(conditioned), &trop_path);
            path = trop_to_log(trop_path);
        }
        auto parameter = project_input(path);
        auto parameter_labels = path_fst_to_labels(parameter).value();
        parameter_labels.erase(
                std::remove_if(
                    parameter_labels.begin(),
                    parameter_labels.end(),
                    [&](Phoneme x) -> bool {
                        return x == phonemes.special("phi");
                    }),
                parameter_labels.end());
        auto parameter_string = labels_to_string(parameter_labels, phonemes);

        lexicon_file << parameter_string << "\t" << observation_string << "\n";
    }

    // std::cout << ctrack::result_as_string();
}
