#ifndef LABELLING_H
#define LABELLING_H

#include <algorithm>
#include <unordered_map>
#include <functional>

template <typename Label, typename Base, typename Special>
class Labelling {
private:
    Label curr_label;
    std::function<Label(Label)> next_label;
    std::unordered_map<Label, Base> map;
    std::unordered_map<Special, Label> specials;
public:
    Labelling(Label first, std::function<Label(Label)> next)
        : curr_label(first), next_label(next) {}
    Label encode(Base base) {
        auto it = std::find_if(
                map.begin(), map.end(),
                [& base](const std::pair<Label, Base>& key){
                    return key.second == base;
                });
        if (it != map.end()) {
            return it->first;
        }
        map[curr_label] = base;
        Label old_label = curr_label;
        curr_label = next_label(curr_label);
        return old_label;
    }
    Label special(Special special) {
        auto it = specials.find(special);
        if (it != specials.end()) {
            return it->second;
        }
        specials[special] = curr_label;
        Label old_label = curr_label;
        curr_label = next_label(curr_label);
        return old_label;
    }
    void associate_special(Special special, Base base) {
        map[specials[special]] = base;
    }
    Base decode(Label label) {
        return map[label];
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

