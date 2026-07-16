#include "impl/gibbs_sampler.hpp"

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
        ) {
    enum class ParameterIndex    : size_t {};
    enum class ObservationIndex    : size_t {};
    enum class ModelIndex : size_t {};

    struct Assignment {
        ParameterIndex parameter;
        ObservationIndex observation;
        std::string split;
    };
    
    std::unordered_map<std::string, double> weights;
    for (const auto& [name, split] : splits) {
        weights[name] = split.mix.boundary;
   }

    // TODO(padril): This gets massive. A custom implementation may be better.
    // TODO(padril): switch to unique_ptr across the repo
    // TODO(padril): several of the new FST(...) new UR(...) new SR(...) lines
    //               would be better if we passed an output fst parameter
    // TODO(padril): might be more explanatory to have Segment as just
    //               Variant<Phoneme, Phone>
    StrongDeque<FST<Log, Phoneme>, ParameterIndex> saved_parameters;
    StrongDeque<FST<Log, Phone>, ObservationIndex> saved_observations;
    StrongDeque<FST<Log, Segment>, ModelIndex> saved_models;
    ModelIndex transform;

    std::vector<Assignment> assignments;
    for (const auto& [alignment, split] : alignments) {
        auto [parameter, observation] = unzip(alignment);

        auto parameter_fst = labels_to_path_fst<Log, Phoneme>(parameter);
        auto parameter_index = ParameterIndex(saved_parameters.size());
        saved_parameters.push_back(parameter_fst);

        auto observation_fst = labels_to_path_fst<Log, Phone>(observation);
        auto observation_index = ObservationIndex(saved_observations.size());
        saved_observations.push_back(observation_fst);
        
        assignments.push_back(Assignment {
                parameter_index,
                observation_index,
                split,
                });
    }

    // TODO(padril): some noexcepts should be added around the codebase

    distribution::MapValueDistribution uniform(
            new distribution::Uniform<Log::ValueType, Log, Engine>(),
            std::function(real_to_log_semiring)
            );

    // TODO(padril): could be std::tuple<Model, SR, UR>
    std::unordered_map<std::pair<ObservationIndex, ParameterIndex>, Log, HashPair>
        likelihood_cache;
    std::unordered_map<std::pair<ParameterIndex, ModelIndex>, ModelIndex, HashPair>
        parameter_model_compose_cache;
    std::unordered_map<std::pair<ModelIndex, ObservationIndex>, ModelIndex, HashPair>
        model_observation_compose_cache;

    // TODO(padril): these lambdas should probably have explicit captures
    //               but right now, they'd be too large. small refactor.

    distribution::MapGivenDistribution<
        ParameterIndex, Log, Engine, ObservationIndex,
        std::pair<std::optional<ParameterIndex>, std::optional<ObservationIndex>>
        > prior(
        new distribution::FSTDistribution<ParameterIndex, Log, Engine, ModelIndex, ParameterIndex, ObservationIndex>(
            &transform,
            [&](Engine& engine, ModelIndex model) -> std::pair<ParameterIndex, Log> {
                auto [alignment, p] = random_walk(engine, saved_models[model]);

                auto parameter = project_input(alignment);
                auto parameter_index = ParameterIndex(saved_parameters.size());
                saved_parameters.push_back(parameter);
                return std::pair(parameter_index, p);
            },
            [&](ParameterIndex parameter, ModelIndex model) -> ModelIndex {
                auto it = parameter_model_compose_cache.find(std::pair(parameter, model));
                if (it != parameter_model_compose_cache.end()) {
                    return it->second;
                }
                auto composed = compose(saved_parameters[parameter], saved_models[model]);
                auto composed_index = ModelIndex(saved_models.size());
                saved_models.push_back(composed);
                parameter_model_compose_cache[std::pair(parameter, model)] = composed_index;
                return composed_index;
            },
            [&](ModelIndex model, ObservationIndex observation) -> ModelIndex {
                auto it = model_observation_compose_cache.find(std::pair(model, observation));
                if (it != model_observation_compose_cache.end()) {
                    return it->second;
                }
                auto composed = compose(saved_models[model], saved_observations[observation]);
                auto composed_index = ModelIndex(saved_models.size());
                saved_models.push_back(composed);
                model_observation_compose_cache[std::pair(model, observation)] = composed_index;
                return composed_index;
            }),
        [](ObservationIndex observation) {
            return std::pair(std::nullopt, observation);
        });

    int likelihood_hits;
    int likelihood_misses;

    std::vector<std::pair<ObservationIndex, ParameterIndex>> dp_initialization;
    for (auto [parameter, observation, split] : assignments) {
        if (splits[split].status != Status::Invisible) {
            dp_initialization.push_back(std::pair(observation, parameter));
        }
    }

    distribution::CRPDirichletProcess<ParameterIndex, Log, Engine, ObservationIndex, Log> dp(
            &prior,
            alpha,
            dp_initialization,
            [&](ObservationIndex observation, ParameterIndex parameter) -> Log {
                auto it = likelihood_cache.find(
                        std::pair(observation, parameter));
                if (it != likelihood_cache.end()) {
                    ++likelihood_hits;
                    return it->second;
                }
                ++likelihood_misses;
                auto l = likelihood(saved_models[transform],
                                    saved_observations[observation],
                                    saved_parameters[parameter]);
                likelihood_cache[std::pair(observation, parameter)] = l;
                return l;
            },
            [&](ObservationIndex observation) -> Log {
                // TODO(padril): cache this as well
                FST<Log, Segment> composed_fst;
                auto it = model_observation_compose_cache.find(
                        std::pair(transform, observation));
                if (it != model_observation_compose_cache.end()) {
                    composed_fst = saved_models[it->second];
                } else {
                    composed_fst = compose(saved_models[transform],
                            saved_observations[observation]);
                    auto composed = ModelIndex(saved_models.size());
                    saved_models.push_back(composed_fst);
                    model_observation_compose_cache[std::pair(transform, observation)] = composed;
                }
                std::vector<Log> dists;
                // TODO(padril): make shortestdistance gracefully exit when no
                //               dists found
                fst::ShortestDistance(composed_fst, &dists, true);
                return static_cast<size_t>(composed_fst.Start()) < dists.size() ?
                    dists[composed_fst.Start()] : Log::Zero();
            },
            [&](ObservationIndex observation, ParameterIndex parameter) -> bool {
                Parameter parameter_form = path_fst_to_labels(saved_parameters[parameter]).value();
                parameter_form.erase(
                        std::remove_if(
                            parameter_form.begin(),
                            parameter_form.end(),
                            [&](Phoneme x) -> bool {
                                return x == phonemes.special("epsilon");
                            }),
                        parameter_form.end());
                Observation observation_form = path_fst_to_labels(saved_observations[observation]).value();
                observation_form.erase(
                        std::remove_if(
                            observation_form.begin(),
                            observation_form.end(),
                            [&](Phone x) -> bool {
                                return x == phones.special("epsilon");
                            }),
                        observation_form.end());
                // TODO(padril): make this a parameter, and do it better
                Log soft_p = real_to_log_semiring(0.01);
                Log soft_sample = uniform.sample(engine, std::monostate()).first;
                // TODO(padril): we can do this with the edit_fst instead
                return levenshtein(observation_form, parameter_form) <= 3
                    || soft_sample <= soft_p;
                // retparametern true;  // Default full-conditional behavioparameter
            },
            &uniform);

    // TODO: This can be abstracted, like Uniform, and moved to distributions.hpp
    class UniformProposal :
        public distribution::Distribution<
            ParameterIndex, std::pair<Log, Log>, Engine, std::pair<ParameterIndex, std::pair<ObservationIndex, size_t>>> {
    private:
        distribution::CRPDirichletProcess<
            ParameterIndex, Log, Engine, ObservationIndex, Log>* dp;
    public:
        UniformProposal(
            distribution::CRPDirichletProcess<
                ParameterIndex, Log, Engine, ObservationIndex, Log>* dp
                ) : dp(dp) {};
        std::pair<ParameterIndex, std::pair<Log, Log>> sample(
                Engine& engine,
                std::pair<ParameterIndex, std::pair<ObservationIndex, size_t>> given
                ) {
            ObservationIndex sr = given.second.first;
            std::vector<ParameterIndex> parameters = dp->get_parameters();
            int n = parameters.size();
            std::uniform_int_distribution<int> uniform_pos{0, n};
            int pos = uniform_pos(engine);
            Log p = real_to_log_semiring(1.0 / n);
            if (pos < n) {
                return std::pair(parameters[pos], std::pair(p, p));
            } else {
                return std::pair(
                        std::get<ParameterIndex>(dp->prior->sample(engine, sr)),
                        std::pair(p, p));
            }
        }
    };

    // For Metropolis-within-Gibbs
    distribution::MetropolisHastings<ParameterIndex, Log, Engine, std::pair<ObservationIndex, size_t>, Log> mh(
            ParameterIndex(0),
            [&](ParameterIndex parameter, std::pair<ObservationIndex, size_t> observation_count) -> Log {
                auto [observation, count] = observation_count;
                auto it = likelihood_cache.find(std::pair(observation, parameter));
                if (it != likelihood_cache.end()) {
                    ++likelihood_hits;
                    return it->second;
                }
                ++likelihood_misses;
                Log l = likelihood(
                        saved_models[transform], saved_observations[observation], saved_parameters[parameter]);
                likelihood_cache[std::pair(observation, parameter)] = l;
                return real_to_log_semiring(count) * l;
            },
            new UniformProposal(&dp),
            &uniform
            );

    // Build edit transducer
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


    {
        size_t n = assignments.size();
        std::vector<std::tuple<FST<Log, Phoneme>, FST<Log, Phone>,
            std::string>> fst_assignments(n);
        for (size_t i = 0; i < n; ++i) {
            auto [parameter, observation, split] = assignments[i];
            fst_assignments[i] = std::tuple(
                    saved_parameters[parameter],
                    saved_observations[observation],
                    split
                    );
        }
        transform = ModelIndex(saved_models.size());
        saved_models.push_back(count_ngrams(fst_assignments, splits, alignemes, edit_fst, 0, self_align));
    }
    std::cout << "\tBuilt initial transform.\n";

    dp_file << "method step i sample nlld\n";
    ur_file << "sample form\n";

    for (size_t step = 0; step < steps; ++step) {
        likelihood_hits = 0;
        likelihood_misses = 0;

        std::cout << "\tStarting step " << step << ".\n";
        // TODO(padril): this is not a good way of doing this
        size_t skipped = 0;
        for (size_t assignment = 0; assignment < assignments.size(); ++assignment) {
            auto [parameter_i, observation_i, split] = assignments[assignment];
            size_t i = assignment - skipped;
            if (splits[split].status != Status::Trainable) {
                ++skipped;
            } else if (uniform.sample(engine, std::monostate()).first < real_to_log_semiring(mh_ratio)) {
                // TODO: this probability is jank
                size_t count = 0;
                std::vector<ParameterIndex> parameters = dp.get_parameters();
                for (ParameterIndex parameter_j : parameters) {
                    if (parameter_i == parameter_j) { ++count; }
                }
                mh.update(parameter_i);
                auto [sample, p] = mh.sample(engine, std::pair(observation_i, count));
                if (parameter_i != sample) {
                    dp.update(i, sample);
                    dp_file << "mh," << step << ',' << i << ',' << size_t(sample)
                        << ',' << p.Value() << '\n';
                }
            } else {
                auto [sample, p] = dp.sample(engine, i);
                dp.update(i, sample);
                mh.update(sample);
                dp_file << "crp," << step << ',' << i << ',' << size_t(sample) << ','
                    << p.Value() << '\n';
            }
        }

        double hit_ratio = static_cast<double>(likelihood_hits)
            / (likelihood_hits + likelihood_misses);
        std::cout << "\t\tLikelihood cache hit ratio : " << hit_ratio << "\n";

        if (step % rebuild_every == rebuild_every - 1) {
            std::cout << "\tRebuilding ngram model\n";
            size_t n = assignments.size();
            std::vector<std::tuple<FST<Log, Phoneme>, FST<Log, Phone>,
                std::string>> fst_assignments(n);
            for (size_t i = 0; i < n; ++i) {
                auto [parameter, observation, split] = assignments[i];
                fst_assignments[i] = std::tuple(
                        saved_parameters[parameter],
                        saved_observations[observation],
                        split
                        );
            }
            transform = ModelIndex(saved_models.size());
            saved_models.push_back(count_ngrams(
                        fst_assignments,
                        splits,
                        alignemes,
                        edit_fst,
                        step,
                        self_align
                        ));
            likelihood_cache.clear();
            parameter_model_compose_cache.clear();
            model_observation_compose_cache.clear();

            std::string model_out_path = model_out_dir + "step_"
                + std::to_string(step) + ".csv";
            std::ofstream model_out(model_out_path);
            model_out << "parameter,observation,split\n";
            for (auto [parameter, observation, split] : fst_assignments) {
                auto parameter_labels = path_fst_to_labels(parameter).value();
                auto parameter_string = labels_to_string(parameter_labels, phonemes);
                auto observation_labels = path_fst_to_labels(observation).value();
                auto observation_string = labels_to_string(observation_labels, phones);
                model_out << parameter_string << ","
                    << observation_string << ","
                    << split << "\n";
            };
        }
    }

    for (size_t i = 0; i < saved_parameters.size(); ++i) {
        std::vector<Phoneme> ur = path_fst_to_labels(saved_parameters[ParameterIndex(i)]).value();
        ur_file << size_t(i) << ',' << labels_to_string(ur, phonemes) << '\n';
    }
}

