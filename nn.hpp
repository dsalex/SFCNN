#pragma once

#include <memory>
#include <vector>

#include "aliases.hpp"
#include "activations.hpp"


class NeuralNetwork {
public:
    ///
    /// \brief NeuralNetwork constructor for a new model
    /// \param layers - describes number of nodes for each layer
    /// \param activation - activation function object
    /// \param rate - learning rate
    ///
    NeuralNetwork(const std::vector<size_t>& layers, Activation* activation, double rate);
    ///
    /// \brief NeuralNetwork constructor from dumped model
    /// \param fname - file name of dumped model
    ///
    NeuralNetwork(const std::string& fname);
    ~NeuralNetwork() {}
    RowVec ForwardProp(const RowVec& x, std::vector<RowVec>& values, std::vector<RowVec>& derivs) const;
    void BackProp(const RowVec& x, const RowVec& y);
    RowVec PredictOne(const RowVec& x) const;
    std::vector<RowVec> Predict(const std::vector<RowVec>& X) const;
private:
    std::vector<Matrix> mNeurons;
    const std::unique_ptr<Activation> mpActivation;
    const Activation& mActivation;
    const double mRate;
};
