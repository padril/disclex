#include "impl/read_model.hpp"
#include "external/rapidcsv.hpp"

std::vector<Alignment> read_model(
        std::vector<std::string> model_paths,
        Labelling<Phoneme, std::string, std::string> phonemes,
        Labelling<Phone, std::string, std::string> phones
        ) {
    std::vector<Alignment> alignments;
    for (auto model_path : model_paths) {
        std::ifstream model_file(model_path, std::ios_base::in);

        rapidcsv::Document doc(model_file, rapidcsv::LabelParams(0, -1));

        std::vector<std::string> observations = doc.GetColumn<std::string>("observation");
        std::vector<std::string> parameters = doc.GetColumn<std::string>("parameter");
        std::vector<std::string> splits = doc.GetColumn<std::string>("split");

        for (size_t i = 0; i < doc.GetRowCount(); ++i) {
            std::vector<Phoneme> ur;
            std::vector<Phone> sr;

            std::istringstream parameter(parameters[i]);
            for (std::string segment; getline(parameter, segment, ' ');) {
                ur.push_back(phonemes.encode(segment));
            }

            std::istringstream observation(observations[i]);
            for (std::string segment; getline(observation, segment, ' ');) {
                sr.push_back(phones.encode(segment));
            }

            // TODO(padril): should error when zip has two args of different length
            alignments.push_back(Alignment {zip(ur, sr), splits[i]});
        }
    }
    return alignments;
}

// split,status,mix
std::unordered_map<std::string, Split> read_splits(std::string splits_path) {
    std::unordered_map<std::string, Split> splits;
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
    return splits;
}
