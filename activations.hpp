#pragma once

#include <armadillo>


class Activation {
public:
    virtual ~Activation()
    {}
    virtual double operator()(const double value) const = 0;
    virtual arma::mat operator()(const arma::mat& value) const = 0;
    virtual double Deriv(double value) const = 0;
    virtual arma::mat Deriv(const arma::mat& value) const = 0;
};

class Sigmoid: public Activation {
public:
    Sigmoid() {}
    ~Sigmoid() {}
    double operator()(const double value) const;
    arma::mat operator()(const arma::mat& value) const;
    double Deriv(const double x) const;
    arma::mat Deriv(const arma::mat& value) const;
};
