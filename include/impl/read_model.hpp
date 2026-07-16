#ifndef READ_MODEL_HPP_
#define READ_MODEL_HPP_

#include "labelling.hpp"
#include "impl/make_model.hpp"

struct Alignment {
    std::vector<std::pair<Phoneme, Phone>> alignment;
    std::string split;
};

std::vector<Alignment> read_model(
        std::vector<std::string> model_paths,
        Labelling<Phoneme, std::string, std::string> phonemes,
        Labelling<Phone, std::string, std::string> phones
        );

std::unordered_map<std::string, Split> read_splits(std::string splits_path);

#endif  // define READ_MODEL_HPP_

