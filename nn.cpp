#include "nn.hpp"

std::vector<arma::mat> NeuralNetwork::Predict(const std::vector<arma::mat>& X) const {
    std::vector<arma::mat> result;

    for (const auto& x: X) {
        auto y = x;
        std::cout << std::endl << y << std::endl;
        for (auto& nm: neurons) {
            y = (*activation)(y*nm);
            std::cout << y << std::endl;
        }
        result.push_back(y);
    }
    return result;
}

