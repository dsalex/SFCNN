#include <iostream>
#include <armadillo>
#include "activations.hpp"
#include "nn.hpp"

using namespace std;

int main() {
    std::vector<size_t> layers{10, 8, 5, 2};

    NeuralNetwork nn(layers, make_activation("sigmoid"), 0.1);
    std::vector<arma::rowvec> X;
    for (size_t i = 0; i < 10; ++i)
        X.push_back(arma::randu<arma::rowvec>(10));
    nn.Predict(X);
    for (size_t i = 0; i < 100; ++i)
        nn.BackProp(arma::randu<arma::rowvec>(10), arma::randu<arma::rowvec>(2));
    return 0;
}

