#ifndef ACTIVATION_FUNCTIONS_H_INCLUDED
#define ACTIVATION_FUNCTIONS_H_INCLUDED

#include <cmath>

inline double sigmoid(double x)
{
    return 1. / (1. + std::exp(-x));
}

inline double sigmoid_deriv(double x)
{
    double s = sigmoid(x);
    return s * (1. - s);
}

inline double relu(double x)
{
    return (x > 0.) ? x : 0.;
}

inline double relu_deriv(double x)
{
    return (x > 0.) ? 1. : 0.;
}

inline double lrelu(double x)
{
    return (x > 0.) ? x : 0.1 * x;
}

inline double lrelu_deriv(double x)
{
    return (x > 0.) ? 1. : 0.1;
}



#endif // ACTIVATION_FUNCTIONS_H_INCLUDED
