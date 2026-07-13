#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <functional>
#include <optional>
#include <variant>
#include <random>
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

template <typename Value, typename Semiring, typename Engine>
class Uniform : public distribution::Distribution<Value, Semiring, Engine, std::monostate> {
private:
    std::uniform_real_distribution<Value> uniform{0.0, 1.0};
public:
    Uniform() {}
    std::pair<Value, Semiring> sample(Engine& engine, std::monostate = std::monostate()) override {
        return std::pair(uniform(engine), Semiring::Zero());
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
        } while (ret.size() < max_length && uniform->sample(engine, std::monostate()).first >= p_stop);
        return std::pair(ret, p);
    }
};

// TODO(padril): this class needs a lot of renaming and formatting
// TODO: a lot of private members are needed across the board
template <
    typename Value,
    typename Semiring,
    typename Engine,
    typename ModelFST,
    typename LeftFST,
    typename RightFST
    >
class FSTDistribution : public Distribution<Value, Semiring, Engine,
    std::pair<std::optional<LeftFST>, std::optional<RightFST>>> {
private:
    using RandomWalkFn = std::function<std::pair<Value, Semiring>(Engine&, ModelFST)>;
    using LeftComposeFn = std::function<ModelFST(LeftFST, ModelFST)>;
    using RightComposeFn = std::function<ModelFST(ModelFST, RightFST)>;
    ModelFST* fst;
    // This is `RandGen` in OpenFST, we'll use a RandGen + path sum
    RandomWalkFn random_walk;
    // This is `Compose` in OpenFST
    LeftComposeFn left_compose;
    RightComposeFn right_compose;
public:
    FSTDistribution(
            ModelFST* fst,
            RandomWalkFn random_walk,
            LeftComposeFn left_compose,
            RightComposeFn right_compose
            ) : fst(fst), random_walk(random_walk), left_compose(left_compose), right_compose(right_compose) {};
    std::pair<Value, Semiring> sample(
            Engine& engine,
            std::pair<std::optional<LeftFST>, std::optional<RightFST>> given) override
    {
        ModelFST fst_ = *fst;
        if (given.first) {
            fst_ = left_compose(given.first.value(), fst_);
        }
        if (given.second) {
            fst_ = right_compose(fst_, given.second.value());
        }
        return random_walk(engine, fst_);
    }
};

// See page 253 of Neal 2000a for how this works, and why certain variables
// are named after greek letters
template <typename Parameter, typename Semiring, typename Engine,
          typename Observation, typename UniformSemiring>
class CRPDirichletProcess : public Distribution<Parameter, Semiring, Engine, size_t> {
public:
    Distribution<Parameter, Semiring, Engine, Observation>* prior;  // G_0
private:
    // TODO: do we have to use a pointer? or is reference fine? i don't remember
    Semiring alpha;
    size_t n;
    // TODO: genericize vector
    std::vector<Observation> observations;  // y
    std::vector<Parameter> parameters;      // theta
    std::function<Semiring(Observation, Parameter)> likelihood;  // F
    std::function<Semiring(Observation)> integrated_likelihood;  // the integral defining r_i
    std::function<bool(Observation, Parameter)> neighbours;  // Not mentioned in Neal, see metric DPs
    Distribution<Semiring, UniformSemiring, Engine, std::monostate>* uniform;
public:
    // TODO: there should be an alternative version that defines integrated
    // likelihood for you, under some circumstances
    CRPDirichletProcess(
            Distribution<Parameter, Semiring, Engine, Observation> *prior,
            Semiring alpha,
            std::vector<std::pair<Observation, Parameter>> initialization,
            std::function<Semiring(Observation, Parameter)> likelihood,
            std::function<Semiring(Observation)> integrated_likelihood,
            std::function<bool(Observation, Parameter)> neighbours,
            Distribution<Semiring, UniformSemiring, Engine, std::monostate>* uniform
            ) : prior(prior),
                alpha(alpha),
                likelihood(likelihood),
                integrated_likelihood(integrated_likelihood),
                neighbours(neighbours),
                uniform(uniform)
    {
        n = initialization.size();
        observations.resize(n);
        parameters.resize(n);
        for (size_t i = 0; i < n; ++i) {
            std::tie(observations[i], parameters[i]) = initialization[i];
        }
    }

    std::pair<Parameter, Semiring> sample(Engine& engine, size_t i) override {
        // TODO: Likelihood temperature
        Semiring r_i = alpha * integrated_likelihood(observations[i]);

        Semiring bound = Semiring::Zero();
        // better than doing a division in the for loop
        Semiring next = uniform->sample(engine, std::monostate()).first * r_i;

        // NOTE: this is possibly optimizable, since our number of parameters
        // will be smaller than the number of indices, and we might be able
        // to store counts and assignments instead
        for (size_t j = 0; j < n; ++j) {
            if (i == j) { continue; }
            // We simply make the assumption that if the observation and
            // parameter are not neighbours, they will have a very low
            // likelihood. TODO(padril): find justification for this in the
            // literature.
            if (!neighbours(observations[i], parameters[j])) { continue; }
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

    void update(size_t i, Parameter theta) {
        parameters[i] = theta;
    }

    std::vector<Parameter> get_parameters() {
        return parameters;
    }
};

template <typename State, typename Semiring, typename Engine, typename Given, typename UniformSemiring>
class MetropolisHastings : public Distribution<State, Semiring, Engine, Given> {
private:
    State state;
    std::function<Semiring(State, Given)> stationary;
    Distribution<State, std::pair<Semiring, Semiring>, Engine, std::pair<State, Given>>* proposal;
    Distribution<Semiring, UniformSemiring, Engine, std::monostate>* uniform;
public:
    MetropolisHastings(
            State initial_state,
            std::function<Semiring(State, Given)> stationary,
            Distribution<State, std::pair<Semiring, Semiring>, Engine, std::pair<State, Given>>* proposal,
            Distribution<Semiring, UniformSemiring, Engine, std::monostate>* uniform
            ) : state(initial_state),
                stationary(stationary),
                proposal(proposal),
                uniform(uniform)
            {}

    std::pair<State, Semiring> sample(Engine& engine, Given given) override {
        State next;
        std::pair<Semiring, Semiring> forward_reverse;
        Semiring forward_p, reverse_p;
        std::tie(next, forward_reverse) =
            proposal->sample(engine, std::pair(state, given));
        std::tie(forward_p, reverse_p) = forward_reverse;
        Semiring curr_p = stationary(state, given);
        Semiring next_p = stationary(next, given);
        Semiring accept_p = std::min(
                Semiring::One(), (next_p / curr_p) * (reverse_p / forward_p));
        Semiring u = uniform->sample(engine, std::monostate()).first;
        if (u <= accept_p) {
            return std::pair(next, forward_p * accept_p);
        } else {
            return std::pair(state, Semiring::One() - forward_p * accept_p);
        }

    }

    void update(State next) {
        state = next;
    }
};

}  // namespace distribution

#endif  // DISTRIBUTION_H

