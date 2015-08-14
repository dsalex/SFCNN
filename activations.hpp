#pragma once

#include <aliases.hpp>


/**
 *  \brief Activation interface class
 */
class Activation {
public:
    virtual ~Activation()
    {}
    virtual double operator()(const double value) const = 0;
    virtual Matrix operator()(const Matrix& value) const = 0;
    virtual double Deriv(double value) const = 0;
    virtual Matrix Deriv(const Matrix& value) const = 0;
};


/**
 *  \brief Sigmoid activation class
 */
class Sigmoid: public Activation {
public:
    Sigmoid() {}
    ~Sigmoid() {}
    double operator()(const double value) const;
    Matrix operator()(const Matrix& value) const;
    double Deriv(const double value) const;
    Matrix Deriv(const Matrix& value) const;
};


/**
 *  \brief Identity activation class
 */
class Identity: public Activation {
public:
    Identity() {}
    ~Identity() {}
    double operator()(const double value) const;
    Matrix operator()(const Matrix& value) const;
    double Deriv(const double value) const;
    Matrix Deriv(const Matrix& value) const;
};


/**
 *  \brief Activation factory
 *  \param[in] actName activation function name, may be "sigmoid", "identity", otherwise exception
 *  \return pointer to activation functor
 */
Activation* make_activation(const std::string& actName);
