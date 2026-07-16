#ifndef LABELLING_H
#define LABELLING_H

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <functional>

template <typename Label, typename Base, typename Special>
class Labelling {
private:
    Label curr_label;
    std::function<Label(Label)> next_label;
    std::unordered_map<Label, Base> map;
    std::unordered_map<Special, Label> specials;
    std::unordered_set<Label> forced;
    Label new_label() {
        Label label = curr_label;
        while (forced.find(label) != forced.end()) {
            label = next_label(label);
        }
        curr_label = next_label(label);
        return label;
    }
public:
    Labelling(Label first, std::function<Label(Label)> next)
        : curr_label(first), next_label(next) {}
    Label encode(const Base& base) {
        auto it = std::find_if(
                map.begin(), map.end(),
                [& base](const std::pair<Label, Base>& key){
                    return key.second == base;
                });
        if (it != map.end()) {
            return it->first;
        }
        Label label = new_label();
        map[label] = base;
        return label;
    }
    Label special(const Special& special) {
        auto it = specials.find(special);
        if (it != specials.end()) {
            return it->second;
        }
        Label label = new_label();
        specials[special] = label;
        return label;
    }
    void associate_special(const Special& special, const Base& base) {
        map[specials[special]] = base;
    }
    void force(const Base& base, const Label& label) {
        map[label] = base;
        forced.insert(label);
    }
    void force_special(const Special& special, const Label& label) {
        specials[special] = label;
        forced.insert(label);
    }
    Base decode(const Label& label) const {
        return map.at(label);
    }
    std::vector<Label> labels() {
        std::vector<Label> ret;
        for (auto& [_, label] : specials) {
            ret.push_back(label);
        }
        for (auto& [label, _] : map) {
            ret.push_back(label);
        }
        return ret;
    }
};

#endif  // ifndef LABELLING_H

