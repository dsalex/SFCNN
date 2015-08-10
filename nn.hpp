#pragma once

#include <memory>
#include <vector>

#include <armadillo>

#include "activations.hpp"

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<size_t> layers, Activation* a, double r = 0.1);
    ~NeuralNetwork() {}
    arma::mat ForwardProp(const arma::mat& x, std::vector<arma::mat> &values, std::vector<arma::mat> &derivs) const;
    void BackProp(const arma::mat& x, const arma::mat& y);
    arma::mat PredictOne(const arma::mat& x) const;
    std::vector<arma::mat> Predict(const std::vector<arma::mat>& X) const;
private:
    const double rate;
    std::vector<arma::mat> neurons;
    std::unique_ptr<Activation> activation;
};
