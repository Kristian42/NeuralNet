#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>


class Layer
{
    public:
        using fun_ptr = double (*)(double);

        Layer();
        Layer(const Eigen::MatrixXd& weights, const Eigen::VectorXd& bias, fun_ptr activation_func, fun_ptr activation_func_deriv);

        Eigen::MatrixXd& getWeight();
        const Eigen::MatrixXd& getWeight() const;

        Eigen::VectorXd& getBias();
        const Eigen::VectorXd& getBias() const;

        void setActivationFunction(fun_ptr activation_func, fun_ptr activation_func_deriv);

        const Eigen::VectorXd& getPreactivation() const;

        Eigen::VectorXd getActivation() const;

        const Eigen::VectorXd& getDelta() const;

        Eigen::VectorXd forwardProp(const Eigen::VectorXd& a_in);
        Eigen::VectorXd backwardProp(const Eigen::VectorXd& input);

    private:
        fun_ptr act_fun;
        fun_ptr act_fun_deriv;
        Eigen::VectorXd z;
        Eigen::MatrixXd W;
        Eigen::VectorXd b;

        Eigen::VectorXd delta;

};

#endif // LAYER_H
