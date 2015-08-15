#include <iostream>
#include <armadillo>
#include "activations.hpp"
#include "nn.hpp"

using namespace std;

int main_bak() {
    std::vector<size_t> layers{10, 8, 5, 3, 2};

    NeuralNetwork nn(layers, make_activation("identity"), 0.1);
    std::vector<arma::rowvec> X;
    for (size_t i = 0; i < 10; ++i)
        X.push_back(arma::randu<arma::rowvec>(10));
    nn.Predict(X);
    for (size_t i = 0; i < 1000; ++i)
        nn.BackProp(arma::randu<arma::rowvec>(10), arma::randu<arma::rowvec>(2));
    nn.SaveModel("dump.txt", true);
    NeuralNetwork nn2("dump.txt", true);
    nn2.SaveModel("dump2.txt", true);
    return 0;
}

int main() {
    std::vector<size_t> layers{64, 10};

    NeuralNetwork nn(layers, make_activation("sigmoid"), 0.01);
    std::ifstream fin("../scripts/digits");
    for (int i = 0; i < 1797; ++i) {
        auto x = RowVec(64);
        for (int xi = 0; xi < 64; ++xi)
            fin >> x[xi];
        auto y = RowVec(10);
        for (int yi = 0; yi < 10; ++yi)
            fin >> y[yi];
        nn.BackProp(x, y);
    }
    nn.SaveModel("dump_digits.txt", true);
    return 0;
}

int main_iris() {
    std::vector<size_t> layers{4, 3};

    NeuralNetwork nn(layers, make_activation("sigmoid"), 0.1);
    std::ifstream fin("../scripts/iris.txt");
    for (int i = 0; i < 150; ++i) {
        auto x = RowVec(4);
        for (int xi = 0; xi < 4; ++xi)
            fin >> x[xi];
        auto y = RowVec(3);
        for (int yi = 0; yi < 3; ++yi)
            fin >> y[yi];
        nn.BackProp(x, y);
    }
    return 0;
}
