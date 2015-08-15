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
    NeuralNetwork(const std::vector<size_t>& layers, std::unique_ptr<Activation> activation, double rate);
    ///
    /// \brief NeuralNetwork constructor from dumped model or config file
    /// \param fname - file name of dumped model
    /// \param isDump - dump or config
    ///
    NeuralNetwork(const std::string& fname, bool isDump);
    ///
    /// \brief Dump
    /// \param fname
    /// \param isDump
    ///
    void SaveModel(const std::string& fname, bool isDump);
    ///
    /// \brief Destructor
    ///
    ~NeuralNetwork() {}
    ///
    /// \brief ForwardProp
    /// \param x
    /// \param values
    /// \param derivs
    /// \return
    ///
    RowVec ForwardProp(const RowVec& x, std::vector<RowVec>& values, std::vector<RowVec>& derivs) const;
    ///
    /// \brief BackProp
    /// \param x
    /// \param y
    ///
    void BackProp(const RowVec& x, const RowVec& y);
    ///
    /// \brief PredictOne
    /// \param x
    /// \return
    ///
    RowVec PredictOne(const RowVec& x) const;
    ///
    /// \brief Predict
    /// \param X
    /// \return
    ///
    std::vector<RowVec> Predict(const std::vector<RowVec>& X) const;
private:
    std::vector<Matrix> mNeurons;
    std::unique_ptr<Activation> mActivation;
    double mRate;
};
