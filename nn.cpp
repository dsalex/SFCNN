#include <cassert>

#include "nn.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& layers, std::unique_ptr<Activation> activation, double rate)
    : mActivation(std::move(activation))
    , mRate(rate)
{
    // Init neurons with [-1/2n; 1/2n] random weights
    for (size_t i = 1; i < layers.size(); ++i) {
        mNeurons.push_back((arma::randu<Matrix>(layers[i-1],layers[i]) - 0.5) / layers[i-1]);
    }
    XSize = layers.front();
    YSize = layers.back();
}

RowVec NeuralNetwork::Predict(const RowVec& x) const {
    auto py = x;
    //std::cout << std::endl << py << std::endl;
    for (auto& nm: mNeurons) {
        py = mActivation->Value(py*nm);
        //std::cout << py << std::endl;
    }
    return py;
}

std::vector<Matrix> NeuralNetwork::BackProp(const RowVec& x, const RowVec& y) {
    std::vector<RowVec> values;
    std::vector<RowVec> derivs;
    std::vector<Matrix> deltas;
    Matrix py = ForwardProp(x, values, derivs);
    Matrix e = py - y;
    // DEBUG
    //std::cout << arma::dot(e, e) << std::endl;
    //std::cout << e << std::endl;
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
        //mNeurons[i-1] -= delta;
        deltas.push_back(std::move(delta));
        //std::cout << "DELTA " << i << " : " << delta << std::endl;
    }
    return deltas;
}

void NeuralNetwork::UpdateWeights(std::vector<Matrix> deltas) {
    assert(deltas.size() == mNeurons.size());
    for (size_t i = 0; i < deltas.size(); ++i) {
        mNeurons[i] -= deltas[i];
    }
}

void NeuralNetwork::SetNeurons(std::vector<Matrix> neurons) {
    assert(neurons.size() == mNeurons.size());
    for (size_t i = 0; i < neurons.size(); ++i) {
        mNeurons[i] = neurons[i];
    }
}


const std::vector<Matrix>& NeuralNetwork::GetNeurons() const {
    return mNeurons;
}

void NeuralNetwork::Train(const RowVec& x, const RowVec& y) {
    UpdateWeights(BackProp(x, y));
}

RowVec NeuralNetwork::ForwardProp(const RowVec& x, std::vector<RowVec>& values, std::vector<RowVec>& derivs) const {
    const Matrix empty;
    values.push_back(x);
    derivs.push_back(empty);
    RowVec py = x;

    for (auto& nm: mNeurons) {
        derivs.push_back(mActivation->Deriv(py*nm));
        py = mActivation->Value(py*nm);
        values.push_back(py);
    }

    return py;
}

void NeuralNetwork::SaveModel(const std::string& fname, bool isDump) {
    if (mNeurons.size() == 0) {
        throw std::logic_error("There are no neurons to dump.");
    }

    std::ofstream f(fname);

    f << mActivation->Name() << std::endl
      << mRate << std::endl
      << mNeurons.size() + 1 << std::endl;

    if (isDump) {
        for (const auto& nm: mNeurons) {
            nm.save(f, arma::arma_ascii);
        }
    } else {
        for (const auto& nm: mNeurons) {
            f << nm.n_rows << " ";
        }
        f << mNeurons.back().n_cols;
    }
}



NeuralNetwork::NeuralNetwork(const std::string& fname, bool isDump)
{
    std::ifstream f(fname);

    std::string act_name;
    f >> act_name;
    mActivation = make_activation(act_name);

    f >> mRate;

    if (isDump) {
        size_t n_layers;
        f >> n_layers;
        mNeurons.resize(n_layers - 1);
        for (auto& nm: mNeurons) {
            nm.load(f, arma::arma_ascii);
        }
    } else {
        std::vector<size_t> layers;
        size_t n_layers;
        f >> n_layers;
        layers.resize(n_layers, 0);
        for (size_t i = 0; i < n_layers; ++i) {
            f >> layers[i];
        }
        // Init neurons with [-1/2n; 1/2n] random weights
        for (size_t i = 1; i < layers.size(); ++i) {
            mNeurons.push_back((arma::randu<Matrix>(layers[i-1],layers[i]) - 0.5) / layers[i-1]);
        }
    }
    XSize = mNeurons.front().n_rows;
    YSize = mNeurons.back().n_cols;
}
