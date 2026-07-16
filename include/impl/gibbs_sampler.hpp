#ifndef GIBBS_SAMPLER_HPP_
#define GIBBS_SAMPLER_HPP_

#include <random>
#include "impl/segment.hpp"
#include "impl/make_model.hpp"
#include "labelling.hpp"
#include "impl/read_model.hpp"

// TODO(padril): badness
using Engine = std::default_random_engine;

// TODO(padril): could be way better, maybe using boost
struct HashPair {
    template <class T, class U>
    std::size_t operator()(const std::pair<T, U> &p) const {
        auto h1 = std::hash<T>{}(p.first);
        auto h2 = std::hash<U>{}(p.second);
        return h1 ^ h2;
    }
};


template <typename T, typename Index>
class StrongDeque {
private:
    std::deque<T> data;
public:
    T& operator[](Index i) {
        return data[static_cast<size_t>(i)];
    }
    const T& operator[](Index i) const {
        return data[static_cast<size_t>(i)];
    }
    void push_back(const T& x) {
        data.push_back(x);
    }
    std::size_t size() const {
        return data.size();
    }
};


// TODO(padril): add static and extern around the codebase
// Slightly modified version of the code from @guilhermeagostinelli on github,
// though this is a standard algorithm.
template <typename T, typename U>
int levenshtein(std::vector<T> form1, std::vector<U> form2) {
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
            int cost = (int(form2[j - 1]) == int(form1[i - 1])) ? 0 : 1;

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


void gibbs_fst_dijkstra(
        size_t steps, Engine engine,
        std::vector<Alignment> alignments,
        std::unordered_map<std::string, Split> splits,
        Labelling<Aligneme, std::pair<Phoneme, Phone>, std::string>& alignemes,
        Labelling<Phoneme, std::string, std::string>& phonemes,
        Labelling<Phone, std::string, std::string>& phones,
        Log alpha,
        size_t rebuild_every,
        bool self_align,
        double mh_ratio,
        std::ofstream& dp_file,
        std::ofstream& ur_file,
        std::string model_out_dir
        );

#endif  // define GIBBS_SAMPLER_HPP_
