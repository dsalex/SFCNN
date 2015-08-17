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
            RowVec delta_y = nn.Predict(x) - y;
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
            std::cout << nn.Predict(x);
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
            nn.Train(x, y);
        }
    }

    if (cmdSave.isSet()) {
        nn.SaveModel(cmdSave.getValue(), true);
    }
    return 0;
}

