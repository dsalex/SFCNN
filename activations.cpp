#include <math.h>

#include "activations.hpp"

double Sigmoid::operator()(const double x) const {
    return 1.0 / (1.0 + exp(-x));
}

arma::mat Sigmoid::operator()(const arma::mat& value) const {
    auto result = value;
    for (int i = 0; i < result.n_rows; ++i)
        for (int j = 0; j < result.n_cols; ++j)
            result(i, j) = (*this)(value(i, j));
    return result;
}

double Sigmoid::Deriv(const double x) const {
    const double sigmoid = (*this)(x);
    return sigmoid * (1 - sigmoid);
}

arma::mat Sigmoid::Deriv(const arma::mat& value) const {
    auto result = value;
    for (int i = 0; i < result.n_rows; ++i)
        for (int j = 0; j < result.n_cols; ++j)
            result(i, j) = this->Deriv(value(i, j));
    return result;
}

