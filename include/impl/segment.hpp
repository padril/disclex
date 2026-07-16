#ifndef SEGMENT_HPP_
#define SEGMENT_HPP_

#include <functional>
#include <iostream>

#include "fst/fstlib.h"

#define DISTINCT_SEGMENT(Name) \
    struct Name { \
        int v; \
        Name(int v = 0) : v(v) {} \
        constexpr explicit operator int() const noexcept { return v; } \
        constexpr explicit operator size_t() const noexcept { return v; } \
        constexpr explicit operator bool() const noexcept { return v; } \
        inline constexpr bool operator==(const Name& other) const noexcept { \
            return v == other.v; \
        } \
        inline constexpr bool operator!=(const Name& other) const noexcept { \
            return v != other.v; \
        } \
        inline constexpr bool operator>(const Name& other) const noexcept { \
            return v > other.v; \
        } \
        inline constexpr bool operator<(const Name& other) const noexcept { \
            return v < other.v; \
        } \
        inline constexpr bool operator<=(const Name& other) const noexcept { \
            return v <= other.v; \
        } \
        inline constexpr bool operator>=(const Name& other) const noexcept { \
            return v >= other.v; \
        } \
        bool Write(std::ostream&) const { return false; } \
    }; \
    namespace std { \
        template <> \
        struct hash<Name> { \
            size_t operator()(const Name& s) const noexcept { \
                return std::hash<int>{}(s.v); \
            } \
        }; \
    } \
    std::ostream& operator<<(std::ostream& os, const Name& name);

template <typename T, typename U>
struct VariantSegment {
    int v;
    VariantSegment(int v = 0) : v(v) {}
    VariantSegment(const T& t) : v(t.v) {}
    VariantSegment(const U& u) : v(u.v) {}
    explicit operator T() const noexcept { return v; }
    explicit operator U() const noexcept { return v; }
    constexpr explicit operator int() const noexcept { return v; }
    constexpr explicit operator size_t() const noexcept { return v; }
    inline constexpr bool operator==(const VariantSegment<T, U>& other) const noexcept {
        return v == other.v;
    }
    inline constexpr bool operator!=(const VariantSegment<T, U>& other) const noexcept {
        return v != other.v;
    }
    inline constexpr bool operator>(const VariantSegment<T, U>& other) const noexcept {
        return v > other.v;
    }
    inline constexpr bool operator<(const VariantSegment<T, U>& other) const noexcept {
        return v < other.v;
    }
    inline constexpr bool operator>=(const VariantSegment<T, U>& other) const noexcept {
        return v >= other.v;
    }
    inline constexpr bool operator<=(const VariantSegment<T, U>& other) const noexcept {
        return v <= other.v;
    }
    bool Write(std::ostream&) const { return false; }
};

namespace std {
    template <typename T, typename U>
    struct hash<VariantSegment<T, U>> {
        size_t operator()(const VariantSegment<T, U>& s) const noexcept {
            return std::hash<int>{}(s.v);
        }
    };
}

DISTINCT_SEGMENT(Phoneme);
DISTINCT_SEGMENT(Phone);
DISTINCT_SEGMENT(Aligneme);
using Segment = VariantSegment<Phoneme, Phone>;

// TODO(padril): should we also abstract std::vector to something else?
// using Variable = std::vector<Segment>;  // In the sense of a random variable
using Observation = std::vector<Phone>;
using Parameter = std::vector<Phoneme>;

#endif  // define SEGMENT_HPP_
