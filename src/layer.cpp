#include "layer.h"

#include <Eigen/Dense>

using namespace Eigen;

Layer::Layer() :
    act_fun(nullptr),
    act_fun_deriv(nullptr),
    z(),
    W(),
    b(),
    delta()
{
    //ctor
}

Layer::Layer(const MatrixXd& weights, const VectorXd& bias, fun_ptr activation_func, fun_ptr activation_func_deriv) :
    act_fun(activation_func),
    act_fun_deriv(activation_func_deriv),
    z(),
    W(weights),
    b(bias),
    delta()
{
    if (weights.rows() != bias.rows())
        throw std::runtime_error("Weights and bias have unequal number of rows.");
}

MatrixXd& Layer::getWeight()
{
    return W;
}

const MatrixXd& Layer::getWeight() const
{
    return W;
}

VectorXd& Layer::getBias()
{
    return b;
}

const VectorXd& Layer::getBias() const
{
    return b;
}

void Layer::setActivationFunction(fun_ptr activation_func, fun_ptr activation_func_deriv)
{
    act_fun = activation_func;
    act_fun_deriv = activation_func_deriv;
}

const VectorXd& Layer::getPreactivation() const
{
    return z;
}

VectorXd Layer::getActivation() const
{
    return z.unaryExpr(act_fun);
}

const VectorXd& Layer::getDelta() const
{
    return delta;
}

VectorXd Layer::forwardProp(const VectorXd& a_in)
{
    z = W * a_in + b;
    return z.unaryExpr(act_fun);
}

VectorXd Layer::backwardProp(const VectorXd& input)
{
    delta = z.unaryExpr(act_fun_deriv).asDiagonal() * input;
    return W.transpose() * delta;
}
