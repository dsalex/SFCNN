#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdexcept>
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <numpy/ndarrayobject.h>

#include "numpyarraydata.h"
#include "aliases.hpp"
#include "nn.hpp"
#include "wrapper.hpp"

void ArmaToNumPy(const std::vector<Matrix>& armas, bp::list& nps) {
    for (const Matrix& armaDelta: armas) {
        const bp::tuple shape = bp::make_tuple(armaDelta.n_rows, armaDelta.n_cols);
        auto npDelta = np::empty(shape, np::dtype::get_builtin<double>());
        NumPyArrayData<double> npDeltaData(npDelta);
        for (size_t row = 0; row < armaDelta.n_rows; ++row) {
            for (size_t col = 0; col < armaDelta.n_cols; ++col) {
                npDeltaData(row, col) = armaDelta(row, col);
            }
        }
        nps.append(npDelta);
    }
}

void NumPyToArma(const bp::list& nps, std::vector<Matrix>& armas) {
    bp::ssize_t n = bp::len(nps);
    for(bp::ssize_t i=0;i<n;i++) {
        np::ndarray npMatrix = bp::extract<np::ndarray>(nps[i]);
        assert(npMatrix.get_nd() == 2);
        assert(npMatrix.get_dtype() == np::dtype::get_builtin<double>());
        Matrix armaMatrix(npMatrix.shape(0), npMatrix.shape(1));
        NumPyArrayData<double> npData(npMatrix);
        for (size_t row = 0; row < armaMatrix.n_rows; ++row) {
            for (size_t col = 0; col < armaMatrix.n_cols; ++col) {
                armaMatrix(row, col) = npData(row, col);
                //std::cout << npData(row, col) << std::endl;
            }
        }
        armas.push_back(armaMatrix);
    }
}


void PyNN::FromConfig(const std::string& fname) {
    delete NN;
    NN = nullptr;
    NN = new NeuralNetwork(fname, false);
}

void PyNN::FromDump(const std::string& fname) {
    delete NN;
    NN = nullptr;
    NN = new NeuralNetwork(fname, true);
}

void PyNN::ToDump(const std::string& fname) {
    NN->SaveModel(fname, true);
}

bp::list PyNN::BackProp(const np::ndarray& x, const np::ndarray& y) {
    assert(x.get_nd() == 1);
    assert(x.get_dtype() == np::dtype::get_builtin<double>());
    assert(y.get_nd() == 1);
    assert(y.get_dtype() == np::dtype::get_builtin<double>());
    assert(NN != nullptr);

    NumPyArrayData<double> xd(x);
    NumPyArrayData<double> yd(y);
    RowVec xa(x.shape(0));
    RowVec ya(y.shape(0));
    for (size_t i = 0; i < x.shape(0); ++i) {
        xa[i] = xd(i);
    }
    for (size_t i = 0; i < y.shape(0); ++i) {
        ya[i] = yd(i);
    }
    std::vector<Matrix> armaDeltas = NN->BackProp(xa, ya);
    bp::list npDeltas;
    ArmaToNumPy(armaDeltas, npDeltas);
    return npDeltas;
}

void PyNN::SetNeurons(const bp::list& neurons) {
    std::vector<Matrix> armaNeurons;
    NumPyToArma(neurons, armaNeurons);
    NN->SetNeurons(armaNeurons);
}


bp::list PyNN::GetNeurons() const {
    std::cout << "<GetNeurons>n";
    bp::list npNeurons;
    ArmaToNumPy(NN->GetNeurons(), npNeurons);
    std::cout << "</GetNeurons>n";
    return npNeurons;
}



BOOST_PYTHON_MODULE(libSFCNN)
{
    using namespace boost::python;
    class_<PyNN>("NN", init<>())
        .def("FromConfig", &PyNN::FromConfig)
        .def("FromDump", &PyNN::FromDump)
        .def("ToDump", &PyNN::ToDump)
        .def("BackProp", &PyNN::BackProp)
        .def("GetNeurons", &PyNN::GetNeurons)
        .def("SetNeurons", &PyNN::SetNeurons)
    ;
}
