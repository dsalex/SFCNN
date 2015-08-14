#include <math.h>

#include "activations.hpp"


double Sigmoid::operator()(const double value) const {
    return 1.0 / (1.0 + exp(-value));
}

Matrix Sigmoid::operator()(const Matrix& value) const {
    auto result = value;
    for (int i = 0; i < result.n_rows; ++i)
        for (int j = 0; j < result.n_cols; ++j)
            result(i, j) = (*this)(value(i, j));
    return result;
}

double Sigmoid::Deriv(const double value) const {
    const double sigmoid = (*this)(value);
    return sigmoid * (1 - sigmoid);
}

Matrix Sigmoid::Deriv(const Matrix& value) const {
    auto result = value;
    for (int i = 0; i < result.n_rows; ++i)
        for (int j = 0; j < result.n_cols; ++j)
            result(i, j) = this->Deriv(value(i, j));
    return result;
}


double Identity::operator()(const double value) const {
    return value;
}

Matrix Identity::operator()(const Matrix& value) const {
    return value;
}

double Identity::Deriv(const double value) const {
    return 1.0;
}

Matrix Identity::Deriv(const Matrix& value) const {
    return arma::ones(value.n_rows, value.n_cols);
}


Activation* make_activation(const std::string& actName) {
    if (actName == "sigmoid") {
        return new Sigmoid();
    } else if (actName == "identity") {
        return new Identity();
    } else {
        throw std::logic_error("Unknown activation function name");
    }
}
