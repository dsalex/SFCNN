#include <cassert>

#include "nn.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& layers, Activation* activation, double rate)
    : mpActivation(activation)
    , mActivation(*activation)
    , mRate(rate)
{
    for (size_t i = 1; i < layers.size(); ++i) {
        mNeurons.push_back(arma::randu<Matrix>(layers[i-1],layers[i]));
    }
}

RowVec NeuralNetwork::PredictOne(const RowVec& x) const {
    auto py = x;
    std::cout << std::endl << py << std::endl;
    for (auto& nm: mNeurons) {
        py = mActivation(py*nm);
        std::cout << py << std::endl;
    }
    return py;
}

std::vector<RowVec> NeuralNetwork::Predict(const std::vector<RowVec>& X) const {
    std::vector<RowVec> result;
    for (const auto& x: X) {
        result.push_back(PredictOne(x));
    }
    return result;
}

void NeuralNetwork::BackProp(const RowVec& x, const RowVec& y) {
    std::vector<RowVec> values;
    std::vector<RowVec> derivs;
    Matrix py = ForwardProp(x, values, derivs);
    Matrix e = py - y;
    // DEBUG
    std::cout << e << std::endl;
    assert(derivs.size() == mNeurons.size() + 1);
    for (size_t i = mNeurons.size(); i > 0; --i) {
        Matrix delta = mNeurons[i-1];
        for (size_t row = 0; row < delta.n_rows; ++row) {
            for (size_t col = 0; col < delta.n_cols; ++col) {
                delta(row, col) = mRate * e(0, col) * derivs[i](col) * values[i-1](row);
            }
        }
        e = e % derivs[i];
        e = e * mNeurons[i-1].t();
        mNeurons[i-1] -= delta;
    }
}

RowVec NeuralNetwork::ForwardProp(const RowVec& x, std::vector<RowVec>& values, std::vector<RowVec>& derivs) const {
    const Matrix fake;
    RowVec py = x;
    derivs.push_back(fake);
    values.push_back(x);

    for (auto& nm: mNeurons) {
        derivs.push_back(mActivation.Deriv(py*nm));
        py = mActivation(py*nm);
        values.push_back(py);
    }

    return py;
}
