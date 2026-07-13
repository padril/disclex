#ifndef IMPL_SEMIRING_H
#define IMPL_SEMIRING_H

#include <fst/fstlib.h>

namespace semiring {

using Log = fst::LogWeightTpl<float>;
using Trop = fst::TropicalWeightTpl<float>;

// TODO(padril): make more things constexpr if you can
constexpr Log real_to_log_semiring(Log::ValueType real) {
    return Log(-std::log(real));
};

}  // namespace semiring

// Doing a lot of operator overloading allows the API to behave better when
// we use the Semirings as we would doubles
namespace fst {

semiring::Log& operator+=(semiring::Log& a, semiring::Log b);
semiring::Log operator+(semiring::Log a, semiring::Log b);

semiring::Log& operator-=(semiring::Log& a, semiring::Log b);
semiring::Log operator-(semiring::Log a, semiring::Log b);

semiring::Log& operator*=(semiring::Log& a, semiring::Log b);
semiring::Log operator*(semiring::Log a, semiring::Log b);

semiring::Log& operator/=(semiring::Log& a, semiring::Log b);
semiring::Log operator/(semiring::Log a, semiring::Log b);

bool operator<(semiring::Log a, semiring::Log b);
bool operator<=(semiring::Log a, semiring::Log b);
bool operator>(semiring::Log a, semiring::Log b);
bool operator>=(semiring::Log a, semiring::Log b);

}  // namespace fst

#endif  // IMPL_SEMIRING_H

