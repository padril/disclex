#ifndef MAKE_MODEL_HPP_
#define MAKE_MODEL_HPP_

#include "impl/semiring.hpp"
using namespace semiring;
#include "impl/fst.hpp"
#include "impl/segment.hpp"

enum class Status {
    Invisible,
    Fixed,
    Trainable,
};

struct Schedule {
    std::vector<std::pair<double, size_t>> points;
    size_t max_step;
    double at_step(size_t step);
};

struct Split {
    Status status;
    Schedule mix;
    size_t order;
};


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

FST<Trop, int> ngrams_from_union(FST<Log, Aligneme> union_, Split split, Aligneme backoff_label);

FST<Log, Segment> count_ngrams(
        std::vector<std::tuple<FST<Log, Phoneme>, FST<Log, Phone>, std::string>> assignments,
        std::unordered_map<std::string, Split> splits,
        Labelling<Aligneme, std::pair<Phoneme, Phone>, std::string>& labelling,
        FST<Log, Segment> edit_fst,
        size_t step,
        bool self_align = false
        );

#endif  // define MAKE_MODEL_HPP_

