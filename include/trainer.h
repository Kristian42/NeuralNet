#ifndef TRAINER_H
#define TRAINER_H

#include "net.h"

#include <Eigen/Dense>

#include <vector>

class Trainer
{
    public:
        using cost_funptr = double (*)(const Eigen::VectorXd&, const Eigen::VectorXd&);
        using deriv_cost_funptr = Eigen::VectorXd (*)(const Eigen::VectorXd&, const Eigen::VectorXd&);
        using Sample = std::pair<Eigen::VectorXd, Eigen::VectorXd>;

        explicit Trainer(Net& network, double step, cost_funptr C, deriv_cost_funptr dC);

        double train(const std::vector<Sample>& data);
        double train(std::vector<Sample>::const_iterator data_begin, std::vector<Sample>::const_iterator data_end, double step, double regul);

        double calculateCost(const Sample& data) const;
        double calculateCost(const std::vector<Sample>& data) const;
        double calculateCostAndGradient(const Sample& data, std::vector<Eigen::MatrixXd>& g_w, std::vector<Eigen::VectorXd>& g_b);

        double calculateRegularization(double strength) const;
        void   addRegularizationGradient(double strength);

    private:
        void updateNet();

        Net& net;
        cost_funptr cost_fun;
        deriv_cost_funptr deriv_cost;
        double stepsize;
        std::vector<Eigen::MatrixXd> grad_W;
        std::vector<Eigen::VectorXd> grad_b;
};

#endif // TRAINER_H
