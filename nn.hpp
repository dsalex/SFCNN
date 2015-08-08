#pragma once

#include <memory>
#include <vector>

#include <armadillo>

#include "activations.hpp"

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<size_t> layers, Activation* a, double r = 0.1)
        : rate(r)
        , activation(a)
    {
        for (size_t i = 1; i < layers.size(); ++i) {
            neurons.push_back(arma::randu<arma::mat>(layers[i-1],layers[i]));
        }
    }
    ~NeuralNetwork() {}
    void Train(std::vector< std::vector<double> >);
    std::vector<arma::mat> Predict(const std::vector<arma::mat> &X) const;
private:
    const double rate;
    std::vector<arma::mat> neurons;
    std::unique_ptr<Activation> activation;
};
