// Copyright (c) Leo Peckham 2026

#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <functional>
#include <utility>
#include <vector>
#include <unordered_map>

#include "labelling.hpp"

#include "impl/make_model.hpp"
#include "impl/gibbs_sampler.hpp"
#include "impl/read_model.hpp"

#include "external/argparse.hpp"
#include <fst/fstlib.h>
#include <fst/extensions/ngram/ngram-fst.h>
#include <ngram/ngram.h>

// #include "external/ctrack.hpp"

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("disclex");

    // TODO(padril): handle unimplemented geometric prior
    program.add_argument("--prior")
        .choices("geometric", "fst")
        .required()
        .help("which type of prior should be used");

    program.add_argument("--alignments")
        .required()
        .append()
        .help("a file containin initial alignment forms. see format "
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
    // program.add_argument("--phi")
    //     .default_value("")
    //     .help("the string identifying phi in phone(me) files, if made "
    //           "explicit");

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
            0,
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
    // std::string phi_ident = program.get<std::string>("--phi");
    // phonemes.special("phi");
    // phones.special("phi");
    // if (phi_ident != "") {
    //     phonemes.associate_special("phi", phi_ident);
    //     phones.associate_special("phi", phi_ident);
    // } else {
    //     phonemes.associate_special("phi", "<phi>");
    //     phones.associate_special("phi", "<phi>");
    // }

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

    std::vector<std::string> alignments_paths = program.get<std::vector<std::string>>("--alignments");
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

    // parameter,observation,split
    std::vector<Alignment> alignments = read_model(
            alignments_paths,
            phonemes,
            phones
            );
    std::cout << "Successfully read alignments file.\n";

    // split,status,mix
    std::unordered_map<std::string, Split> splits = read_splits(splits_path, steps);
    std::cout << "Successfully read splits file.\n";

    Labelling<Aligneme, std::pair<Phoneme, Phone>, std::string> alignemes(
            0,
            [](Aligneme label) -> Aligneme {
                return int(label) + 1;
            });
    // TODO(padril): Move to native FST Symbol maps
    alignemes.force_special("epsilon", 0);
    alignemes.associate_special("epsilon",
            std::pair(phonemes.special("epsilon"), phones.special("epsilon")));
    // alignemes.special("phi");
    // alignemes.associate_special("phi",
    //         std::pair(phonemes.special("phi"), phones.special("phi")));
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

