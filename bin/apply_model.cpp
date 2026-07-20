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

    program.add_argument("--epsilon")
        .default_value("")
        .help("the string identifying epsilon in phone(me) files, if made "
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
    program.add_argument("--max-step")
        .scan<'d', size_t>()
        .help("what step the training went up to (used for mix portions)");
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
    if (epsilon_ident == "") { epsilon_ident = "<eps>"; }
    phonemes.force_special("epsilon", 0);
    phones.force_special("epsilon", 0);
    phonemes.associate_special("epsilon", epsilon_ident);
    phones.associate_special("epsilon", epsilon_ident);

    std::string output_lexicon_path = program.get<std::string>(
            "--output-lexicon");

    // TODO(padril): safely check that each arg is in a reasonable range
    // TODO(padril): safely handle opening files

    std::string observations_path = program.get<std::string>("--observations");
    std::vector<std::string> model_paths = program.get<std::vector<std::string>>("--model");
    std::string splits_path = program.get<std::string>("--splits");

    size_t max_step = program.get<size_t>("--max-step");
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
    std::vector<Alignment> observations_alignments = read_model(
            { observations_path, },
            phonemes,
            phones
            );

    std::cout << "phonemes:";
    for (auto phoneme : phonemes.labels()) {
        std::cout << " " << phoneme.v << ":" << phonemes.decode(phoneme);
    };
    std::cout << "\n";
    std::cout << "phones:";
    for (auto phone : phones.labels()) {
        std::cout << " " << phone.v << ":" << phones.decode(phone);
    };
    std::cout << "\n";

    std::cout << "Successfully read model file.\n";

    std::unordered_map<std::string, Split> splits = read_splits(splits_path, max_step);
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
    alignemes.force_special("epsilon", 0);
    alignemes.associate_special("epsilon",
            std::pair(phonemes.special("epsilon"), phones.special("epsilon")));

    FST<Log, Segment> model = count_ngrams(
            assignments,
            splits,
            alignemes,
            edit_fst,
            step
            );
    std::cout << "Successfully built model.\n";

    std::ofstream lexicon_file(output_lexicon_path, std::ios_base::out);
    lexicon_file << "parameter,observation,split\n";
    for (auto [alignment_labels, split] : observations_alignments ) {
        std::vector<Phone> observation_labels = unzip(alignment_labels).second;
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
                        return x == phonemes.special("epsilon");
                    }),
                parameter_labels.end());
        auto parameter_string = labels_to_string(parameter_labels, phonemes);
        auto observation_string = labels_to_string(observation_labels, phones);

        lexicon_file << parameter_string << ","
            << observation_string << ","
            << split << "\n";
    }

    // std::cout << ctrack::result_as_string();
}
