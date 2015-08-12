#pragma once

#include <aliases.hpp>


class Activation {
public:
    virtual ~Activation()
    {}
    virtual double operator()(const double value) const = 0;
    virtual Matrix operator()(const Matrix& value) const = 0;
    virtual double Deriv(double value) const = 0;
    virtual Matrix Deriv(const Matrix& value) const = 0;
};


class Sigmoid: public Activation {
public:
    Sigmoid() {}
    ~Sigmoid() {}
    double operator()(const double value) const;
    Matrix operator()(const Matrix& value) const;
    double Deriv(const double x) const;
    Matrix Deriv(const Matrix& value) const;
};


class Identity: public Activation {
public:
    Identity() {}
    ~Identity() {}
    double operator()(const double value) const;
    Matrix operator()(const Matrix& value) const;
    double Deriv(const double x) const;
    Matrix Deriv(const Matrix& value) const;
};


Activation* make_activation(const std::string& actName);
