#ifndef SEGMENT_H
#define SEGMENT_H

#include <vector>
#include <iostream>
#include <tuple>
#include <functional>
#include <algorithm>

namespace segment {

template <typename Value>
class Segment {
public:
    virtual ~Segment() = default;
    virtual std::vector<Value> values() = 0;
    virtual size_t size() { return this->values().size(); }

};

// TODO: lowk rethink textmapped as a whole. probably an easier way to do this whole thing.
template <typename Value, typename Text>
class TextMappedSegment : public Segment<Value> {
public:
    // TODO: generalize around getline
    static std::vector<std::tuple<Value, Text>> parse(
            std::wistream& stream,
            std::function<Value(const Text&)> parse_value) {
        std::vector<std::tuple<Value, Text>> ret = {};
        for (Text text; std::getline(stream, text);) {
            Value value = parse_value(text);
            ret.push_back({value, text});
        }
        return ret;
    }


    virtual Text isomorphism(Value value) = 0;
    virtual Value isomorphism(Text value) = 0;

    // TODO: Get a linter for the project
};

template <typename In, typename Out, typename Text>
class AlignmentSegment : public TextMappedSegment<std::pair<In, Out>, Text> {
public:
    In in;
    Out out;
    Text delim;
    AlignmentSegment(In in, Out out, Text delim)
        : in(in), out(out), delim(delim) {}

};

// TODO: we can probably generalize beyond vector
template<typename In, typename Out, typename Text> using
    AlignmentString = std::vector<AlignmentSegment<In, Out, Text>>;

template<typename In, typename Text>
AlignmentString<In, In, Text> identity(std::vector<In> input, Text delim) {
    AlignmentString<In, In, Text> ret(input.size());
    std::transform(input.begin(), input.end(), ret.begin(), [delim](In in) {
            return AlignmentSegment(in, in, delim);
            });
    return ret;
}

} // namespace segment

// TODO: go through and format these all
#endif  // ifndef SEGMENT_H
