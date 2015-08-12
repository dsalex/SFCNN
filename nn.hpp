#pragma once

#include <memory>
#include <vector>

#include "aliases.hpp"
#include "activations.hpp"


class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<size_t>& layers, Activation* activation, double rate = 0.1);
    NeuralNetwork(const std::string& fname);
    ~NeuralNetwork() {}
    RowVec ForwardProp(const RowVec& x, std::vector<RowVec> &values, std::vector<RowVec> &derivs) const;
    void BackProp(const RowVec &x, const RowVec &y);
    RowVec PredictOne(const RowVec& x) const;
    std::vector<RowVec> Predict(const std::vector<RowVec>& X) const;
private:
    std::vector<Matrix> mNeurons;
    const std::unique_ptr<Activation> mpActivation;
    const Activation& mActivation;
    const double mRate;
};
