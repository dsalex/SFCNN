#pragma once

#include <armadillo>


class Activation {
public:
    virtual ~Activation()
    {}
    virtual arma::mat operator()(const arma::mat& value) const = 0;
//    virtual arma::mat Deriv(arma::mat) = 0;
};

class Sigmoid: public Activation {
public:
    Sigmoid() {}
    ~Sigmoid() {}
    arma::mat operator()(const arma::mat& value) const;
};
