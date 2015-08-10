#include <cassert>

#include "nn.hpp"

NeuralNetwork::NeuralNetwork(std::vector<size_t> layers, Activation* a, double r)
    : rate(r)
    , activation(a)
{
    for (size_t i = 1; i < layers.size(); ++i) {
        neurons.push_back(arma::randu<arma::mat>(layers[i-1],layers[i]));
    }
}

arma::mat NeuralNetwork::PredictOne(const arma::mat& x) const {
    auto py = x;
    std::cout << std::endl << py << std::endl;
    for (auto& nm: neurons) {
        py = (*activation)(py*nm);
        std::cout << py << std::endl;
    }
    return py;
}

std::vector<arma::mat> NeuralNetwork::Predict(const std::vector<arma::mat>& X) const {
    std::vector<arma::mat> result;
    for (const auto& x: X) {
        result.push_back(PredictOne(x));
    }
    return result;
}

void NeuralNetwork::BackProp(const arma::mat& x, const arma::mat& y) {
    std::vector<arma::mat> values;
    std::vector<arma::mat> derivs;
    arma::mat py = ForwardProp(x, values, derivs);
    arma::mat e = py - y;
    assert(derivs.size() == neurons.size() + 1);
    for (size_t i = neurons.size(); i > 0; --i) {
        arma::mat delta = neurons[i-1];
        for (size_t row = 0; row < delta.n_rows; ++row)
            for (size_t col = 0; col < delta.n_cols; ++col)
                delta(row, col) = rate * e(0, col) * derivs[i](col) * values[i-1](row);

        e = e % derivs[i];
        e = e * neurons[i-1].t();
        neurons[i-1] -= delta;
    }
}


arma::mat NeuralNetwork::ForwardProp(const arma::mat& x, std::vector<arma::mat>& values, std::vector<arma::mat>& derivs) const {
    const arma::mat fake;
    auto py = x;
    derivs.push_back(fake);
    values.push_back(x);

    for (auto& nm: neurons) {
        derivs.push_back(activation->Deriv(py*nm));
        py = (*activation)(py*nm);
        values.push_back(py);
    }
    return py;
}
