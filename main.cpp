#include <iostream>
#include <armadillo>
#include <tclap/CmdLine.h>
#include "activations.hpp"
#include "nn.hpp"

using namespace std;

int main(int argc, char** argv) {
    TCLAP::CmdLine cmd("Simple fully connected neural network tool", ' ', "0.1");

    TCLAP::ValueArg<std::string> cmdConfig("c", "config", "Config file path", true, "", "path");
    TCLAP::ValueArg<std::string> cmdModel("m", "model", "Load model from file", true, "", "path");
    cmd.xorAdd(cmdConfig, cmdModel);
    TCLAP::ValueArg<std::string> cmdSave("s", "save", "Save model to path", false, "", "path");
    cmd.add(cmdSave);
    TCLAP::SwitchArg cmdTest("t","test","Test model", false);
    TCLAP::SwitchArg cmdPredict("p","predict","Make predictions", false);
    cmd.add(cmdTest);
    cmd.add(cmdPredict);
    cmd.parse(argc, argv);

    std::string fname;
    bool isDump;
    if (cmdConfig.isSet()) {
        fname = cmdConfig.getValue();
        isDump = false;
    } else {
        fname = cmdModel.getValue();
        isDump = true;
    }
    NeuralNetwork nn(fname, isDump);

    if (cmdTest.getValue()) { // Testing mode
        std::string line;
        double err_sum = 0.0;
        int count = 0;
        while (std::getline(std::cin, line)) {
            std::istringstream iss(line);
            RowVec x(nn.XSize);
            RowVec y(nn.YSize);
            for (size_t i = 0; i < nn.XSize; ++i) {
                iss >> x[i];
            }
            for (size_t i = 0; i < nn.YSize; ++i) {
                iss >> y[i];
            }
            RowVec delta_y = nn.PredictOne(x) - y;
            err_sum += arma::dot(delta_y, delta_y);
            count += 1;
        }
        std::cout << "RMSE like: " << sqrt(err_sum / count) << std::endl;
        std::cout << "Per output RMSE: " << sqrt(err_sum / count / nn.YSize) << std::endl;

    } else if (cmdPredict.getValue()) { // Make predictions
        std::string line;
        while (std::getline(std::cin, line))
        {
            std::istringstream iss(line);
            RowVec x(nn.XSize);
            for (size_t i = 0; i < nn.XSize; ++i) {
                iss >> x[i];
            }
            std::cout << nn.PredictOne(x);
        }
    } else { // Training mode (default)
        std::string line;
        while (std::getline(std::cin, line))
        {
            std::istringstream iss(line);
            RowVec x(nn.XSize);
            RowVec y(nn.YSize);
            for (size_t i = 0; i < nn.XSize; ++i) {
                iss >> x[i];
            }
            for (size_t i = 0; i < nn.YSize; ++i) {
                iss >> y[i];
            }
            nn.BackProp(x, y);
        }
    }

    if (cmdSave.isSet()) {
        nn.SaveModel(cmdSave.getValue(), true);
    }
    return 0;
}

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

int main_digits() {
    std::vector<size_t> layers{64, 10};

    NeuralNetwork nn(layers, make_activation("sigmoid"), 0.001);
    for (int k = 0; k < 10; ++k) {
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
