#include <math.h>

#include "activations.hpp"

arma::mat Sigmoid::operator()(const arma::mat& value) const {
    auto result = value;
    for (int i = 0; i < result.n_rows; ++i)
        for (int j = 0; j < result.n_cols; ++j)
            result(i, j) = 1.0 / (1 + exp(-result(i, j)));
    return result;
}
