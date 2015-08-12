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

    std::vector<size_t> layers{10, 8, 5, 2};

    NeuralNetwork nn(layers, new Sigmoid(), 0.01);
    std::vector<arma::rowvec> X;
    for (size_t i = 0; i < 10; ++i)
        X.push_back(arma::randu<arma::rowvec>(10));
    nn.Predict(X);
    for (size_t i = 0; i < 100; ++i)
        nn.BackProp(arma::randu<arma::rowvec>(10), arma::randu<arma::rowvec>(2));
    return 0;
}

