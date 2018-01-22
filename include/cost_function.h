#ifndef COST_FUNCTION_H_INCLUDED
#define COST_FUNCTION_H_INCLUDED

#include <Eigen/Dense>

inline Eigen::VectorXd softmax(const Eigen::VectorXd& x)
{
    Eigen::VectorXd p = x.array().exp();
    return p / p.sum();
}

inline double softmax_cost(const Eigen::VectorXd& act, const Eigen::VectorXd& y)
{
    Eigen::VectorXd q = -softmax(act).array().log();
    return y.transpose() * q;
}

inline Eigen::VectorXd softmax_cost_deriv(const Eigen::VectorXd& act, const Eigen::VectorXd& y)
{
    return softmax(act) - y;
}


#endif // COST_FUNCTION_H_INCLUDED
