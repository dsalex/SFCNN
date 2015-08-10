#include <iostream>
#include <armadillo>
#include "activations.hpp"
#include "nn.hpp"

using namespace std;

int main() {
    arma::mat A = arma::randu<arma::mat>(1,3);
    arma::mat B = arma::randu<arma::mat>(3,5);
    cout << A.n_rows << " " << A.n_cols << "\n";

    cout << A*B << endl;

    std::vector<size_t> layers;
    layers.push_back(10);
    layers.push_back(8);
    layers.push_back(5);
    layers.push_back(2);
    NeuralNetwork nn(layers, new Sigmoid());
    std::vector<arma::mat> X;
    for (size_t i = 0; i < 10; ++i)
        X.push_back(arma::randu<arma::mat>(1,10));
    nn.Predict(X);
    nn.BackProp(arma::randu<arma::mat>(1,10), arma::randu<arma::mat>(1,2));
    nn.BackProp(arma::randu<arma::mat>(1,10), arma::randu<arma::mat>(1,2));
    nn.BackProp(arma::randu<arma::mat>(1,10), arma::randu<arma::mat>(1,2));
    nn.BackProp(arma::randu<arma::mat>(1,10), arma::randu<arma::mat>(1,2));
    return 0;
}

