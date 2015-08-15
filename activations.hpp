#pragma once

#include <memory>

#include <aliases.hpp>


/**
 *  \brief Activation interface class
 */
class Activation {
public:
    virtual ~Activation()
    {}
    virtual const std::string& Name() = 0;
    virtual double Value(const double value) const = 0;
    virtual Matrix Value(const Matrix& value) const = 0;
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
    const std::string& Name() { return mName; }
    double Value(const double value) const;
    Matrix Value(const Matrix& value) const;
    double Deriv(const double value) const;
    Matrix Deriv(const Matrix& value) const;
private:
    static const std::string mName;
};


/**
 *  \brief Identity activation class
 */
class Identity: public Activation {
public:
    Identity() {}
    ~Identity() {}
    const std::string& Name() { return mName; }
    double Value(const double value) const;
    Matrix Value(const Matrix& value) const;
    double Deriv(const double value) const;
    Matrix Deriv(const Matrix& value) const;
private:
    static const std::string mName;
};


/**
 *  \brief Activation factory
 *  \param[in] actName activation function name, may be "sigmoid", "identity", otherwise exception
 *  \return pointer to activation functor
 */
std::unique_ptr<Activation> make_activation(const std::string& actName);
