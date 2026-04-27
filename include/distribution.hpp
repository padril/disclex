#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <functional>
#include <optional>
#include <variant>
// TODO: Across the board, figure out what to mark as final

namespace distribution {

// TODO: Value could probably be more intelligently renamed... in main too
template <typename Value, typename Semiring, typename Engine, typename Given>
class Distribution {
public:
    virtual ~Distribution() = default;
    virtual std::pair<Value, Semiring> sample(Engine& engine, Given given) = 0;
};


// Distributions are functors of their Values
template <typename ToValue, typename Semiring, typename Engine, typename Given, typename FromValue>
class MapValueDistribution : public Distribution<ToValue, Semiring, Engine, Given> {
private:
    Distribution<FromValue, Semiring, Engine, Given> *from;
    std::function<ToValue(FromValue)> convert;
public:
    MapValueDistribution(
            Distribution<FromValue, Semiring, Engine, Given> *from,
            std::function<ToValue(FromValue)> convert
            ) : from(from), convert(convert) {};
    std::pair<ToValue, Semiring> sample(Engine& engine, Given given) override {
        FromValue s;
        Semiring p;
        std::tie(s, p) = from->sample(engine, given);
        return std::pair(convert(s), p);
    }
};

template <typename Value, typename ToSemiring, typename Engine, typename Given, typename FromSemiring>
class MapSemiringDistribution : public Distribution<Value, ToSemiring, Engine, Given> {
private:
    Distribution<Value, FromSemiring, Engine, Given> *from;
    std::function<ToSemiring(FromSemiring)> convert;
public:
    MapSemiringDistribution(
            Distribution<Value, FromSemiring, Engine, Given> *from,
            std::function<ToSemiring(FromSemiring)> convert
            ) : from(from), convert(convert) {};
    std::pair<Value, ToSemiring> sample(Engine& engine, Given given) override {
        Value s;
        FromSemiring p;
        std::tie(s, p) = from->sample(engine, given);
        return std::pair(s, convert(p));
    }
};


// Distributions are inverse functors of their Engines
template <typename Value, typename Semiring, typename ToEngine, typename Given, typename FromEngine>
class MapEngineDistribution : public Distribution<Value, Semiring, ToEngine, Given> {
private:
    Distribution<Value, Semiring, FromEngine, Given> *from;
    std::function<FromEngine&(ToEngine&)> convert;
public:
    MapEngineDistribution(
            Distribution<Value, Semiring, FromEngine, Given> *from,
            std::function<FromEngine(ToEngine)> convert
            ) : from(from), convert(convert) {};
    std::pair<Value, Semiring> sample(ToEngine& engine, Given given) override {
        return from->sample(convert(engine), given);
    }
};

// Distributions are inverse functors of their Givens
template <typename Value, typename Semiring, typename Engine, typename ToGiven, typename FromGiven>
class MapGivenDistribution : public Distribution<Value, Semiring, Engine, ToGiven> {
private:
    Distribution<Value, Semiring, Engine, FromGiven> *from;
    std::function<FromGiven(ToGiven)> convert;
public:
    MapGivenDistribution(
            Distribution<Value, Semiring, Engine, FromGiven> *from,
            std::function<FromGiven(ToGiven)> convert
            ) : from(from), convert(convert) {};
    std::pair<Value, Semiring> sample(Engine& engine, ToGiven given) override {
        return from->sample(engine, convert(given));
    }
};

// TODO: type erasure to make std::vector any iterator
// TODO: is there something a little more generic than std::vector across the board?
template <typename Value, typename Semiring, typename Engine, typename Given>
class Geometric : public Distribution<std::vector<Value>, Semiring, Engine, Given> {
private:
    Distribution<Value, Semiring, Engine, Given>* base;
    Semiring p_stop;
    size_t max_length;
    // TODO: Should this be a pointer?
    Distribution<Semiring, std::monostate, Engine, std::monostate>* uniform;
public:
    Geometric(
            Distribution<Value, Semiring, Engine, Given> *base,
            Semiring p_stop,
            size_t max_length,
            Distribution<Semiring, std::monostate, Engine, std::monostate>* uniform
            ) : base(base), p_stop(p_stop), max_length(max_length), uniform(uniform) {};
    std::pair<std::vector<Value>, double> sample(Engine& engine, Given given) {
        std::vector<Value> ret;
        // TODO: I don't think this p is correct
        Semiring p = Semiring::One();
        do {
            // TODO: This is wrong, should be std::tie(s, p_i) = base->sample
            ret.push_back(base->sample(engine, given));
            p *= Semiring::One() - p_stop;
        } while (ret.size() < max_length && std::get<Semiring>(uniform->sample(engine, std::monostate())) >= p_stop);
        return std::pair(ret, p);
    }
};

// TODO: a lot of private members are needed across the board
template <
    typename Value,
    typename Semiring,
    typename Engine,
    typename FST
    >
class FSTDistribution : public Distribution<Value, Semiring, Engine, std::pair<std::optional<FST>, std::optional<FST>>> {
private:
    using RandomWalkFn = std::function<std::pair<Value, Semiring>(Engine&, FST)>;
    using ComposeFn = std::function<FST(FST, FST)>;
    FST* fst;
    // This is `RandGen` in OpenFST, we'll use a RandGen + path sum
    RandomWalkFn random_walk;
    // This is `Compose` in OpenFST
    ComposeFn compose;
public:
    FSTDistribution(
            FST* fst,
            RandomWalkFn random_walk,
            ComposeFn compose
            ) : fst(fst), random_walk(random_walk), compose(compose) {};
    std::pair<Value, Semiring> sample(
            Engine& engine,
            std::pair<std::optional<FST>, std::optional<FST>> given) override
    {
        FST fst_ = *fst;
        if (given.first) {
            fst_ = compose(given.first.value(), fst_);
        }
        if (given.second) {
            fst_ = compose(fst_, given.second.value());
        }
        return random_walk(engine, fst_);
    }
};

// See page 253 of Neal 2000a for how this works, and why certain variables
// are named after greek letters
template <
    typename Parameter, typename Semiring, typename Engine, typename Index,
    typename Observation, template <typename, typename> typename Map
    >
class GibbsDirichletProcess : public Distribution<Parameter, Semiring, Engine, Index> {
public:
private:
    // TODO: do we have to use a pointer? or is reference fine? i don't remember
    Distribution<Parameter, Semiring, Engine, Observation>* prior;  // G_0
    Semiring alpha;
    // TODO: make Map a better generic
    // TODO: pair these into std::pair<Observation, Parameter>
    Map<Index, Observation> observations;  // y
    Map<Index, Parameter> parameters;      // theta
    std::function<Semiring(Observation, Parameter)> likelihood;  // F
    std::function<Semiring(Observation)> integrated_likelihood;  // the integral defining r_i
    Distribution<Semiring, std::monostate, Engine, std::monostate>* uniform;
public:
    // TODO: there should be an alternative version that defines integrated
    // likelihood for you, under some circumstances
    GibbsDirichletProcess(
            Distribution<Parameter, Semiring, Engine, Observation> *prior,
            Semiring alpha,
            Map<Index, Observation> observations,
            Map<Index, Parameter> initial_parameters,
            std::function<Semiring(Observation, Parameter)> likelihood,
            std::function<Semiring(Observation)> integrated_likelihood,
            Distribution<Semiring, std::monostate, Engine, std::monostate>* uniform
            ) : prior(prior),
                alpha(alpha),
                observations(observations),
                parameters(initial_parameters),
                likelihood(likelihood),
                integrated_likelihood(integrated_likelihood),
                uniform(uniform)
            {}

    std::pair<Parameter, Semiring> sample(Engine& engine, Index i) override {
        Semiring r_i = alpha * integrated_likelihood(observations[i]);

        Semiring bound = Semiring::Zero();
        // better than doing a division in the for loop
        Semiring next = std::get<Semiring>(uniform->sample(engine, std::monostate())) * r_i;

        // NOTE: this is possibly optimizable, since our number of parameters
        // will be smaller than the number of indices, and we might be able
        // to store counts and assignments instead
        for (Index j = 0; j < (Index) parameters.size(); ++j) {
            if (i == j) { continue; }
            Semiring q_i_j = likelihood(observations[i], parameters[j]);
            bound += q_i_j;
            if (next <= bound) {
                return std::pair(parameters[j], q_i_j / r_i);
            }
        }
        // NOTE: we can ignore the calculation of b because of this being
        // essentially an "otherwise", so we don't need to add up to 1
        Parameter theta;
        Semiring p;
        std::tie(theta, p) = prior->sample(engine, observations[i]);
        return std::pair(theta, p * (Semiring::One() - bound));
    }

    void update(Index i, Parameter theta) {
        parameters[i] = theta;
    }

    Map<Index, Parameter> get_parameters() {
        return parameters;
    }
};

}  // namespace distribution

#endif  // DISTRIBUTION_H

