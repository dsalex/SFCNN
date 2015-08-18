#pragma once

namespace bp = boost::python;
namespace np = boost::numpy;

class PyNN {
public:
    PyNN()
        : NN(nullptr)
    {}
    ~PyNN(){
        delete NN;
    }
    void FromConfig(const std::string& fname);
    void FromDump(const std::string& fname);
    void ToDump(const std::string& fname);
    bp::list BackProp(const np::ndarray& x, const np::ndarray& y);
    void SetNeurons(const bp::list& neurons);
    bp::list GetNeurons() const;
private:
    NeuralNetwork* NN;
};
