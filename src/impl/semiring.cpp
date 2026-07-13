#include "impl/semiring.hpp"

namespace fst {

semiring::Log& operator+=(semiring::Log& a, semiring::Log b) {
    a = Plus(a, b);
    return a;
}

semiring::Log operator+(semiring::Log a, semiring::Log b) {
    a += b;
    return a;
}

semiring::Log& operator-=(semiring::Log& a, semiring::Log b) {
    a = Minus(a, b);
    return a;
}

semiring::Log operator-(semiring::Log a, semiring::Log b) {
    a -= b;
    return a;
}

semiring::Log& operator*=(semiring::Log& a, semiring::Log b) {
    a = Times(a, b);
    return a;
}

semiring::Log operator*(semiring::Log a, semiring::Log b) {
    a *= b;
    return a;
}

semiring::Log& operator/=(semiring::Log& a, semiring::Log b) {
    a = Divide(a, b);
    return a;
}

semiring::Log operator/(semiring::Log a, semiring::Log b) {
    a /= b;
    return a;
}

bool operator<(semiring::Log a, semiring::Log b) {
    return a.Value() > b.Value();
}

bool operator<=(semiring::Log a, semiring::Log b) {
    return a < b || a == b;
}

bool operator>(semiring::Log a, semiring::Log b) {
    return !(a < b);
}

bool operator>=(semiring::Log a, semiring::Log b) {
    return a > b || a == b;
}

};  // namespace fst
