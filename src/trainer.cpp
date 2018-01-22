#include "trainer.h"
#include "net.h"

#include <vector>

Trainer::Trainer(Net& network, double step, cost_funptr C, deriv_cost_funptr dC) :
    net(network),
    cost_fun(C),
    deriv_cost(dC),
    stepsize(step),
    grad_W(),
    grad_b()
{
    //ctor
}

double Trainer::train(const std::vector<Sample>& data)
{
    const int num_samples = data.size();

    std::vector<Eigen::MatrixXd> delta_grad_W;
    std::vector<Eigen::VectorXd> delta_grad_b;

    double C = calculateCostAndGradient(data.front(), grad_W, grad_b) / num_samples;
    for (auto& g : grad_W)
        g /= num_samples;
    for (auto& g : grad_b)
    {
        g /= num_samples;
    }

    for (auto it = data.begin() + 1; it != data.end(); ++it)
    {
        C += calculateCostAndGradient(*it, delta_grad_W, delta_grad_b) / num_samples;
        for (size_t i = 0; i < delta_grad_W.size(); ++i)
        {
            grad_W[i] += delta_grad_W[i] / num_samples;
            grad_b[i] += delta_grad_b[i] / num_samples;
        }
    }

    updateNet();

    return C;
}

double Trainer::train(std::vector<Sample>::const_iterator beg, std::vector<Sample>::const_iterator end, double step, double regul)
{
    const std::vector<Sample>::difference_type num_samples = end - beg;

    std::vector<Eigen::MatrixXd> delta_grad_W;
    std::vector<Eigen::VectorXd> delta_grad_b;

    auto iter = beg;

    double C = calculateCostAndGradient(*iter, grad_W, grad_b) / num_samples;
    for (auto& g : grad_W)
        g /= num_samples;
    for (auto& g : grad_b)
        g /= num_samples;

    while (iter != end)
    {
        C += calculateCostAndGradient(*iter, delta_grad_W, delta_grad_b) / num_samples;
        for (size_t i = 0; i < delta_grad_W.size(); ++i)
        {
            grad_W[i] += delta_grad_W[i] / num_samples;
            grad_b[i] += delta_grad_b[i] / num_samples;
        }
        ++iter;
    }

    addRegularizationGradient(regul);

    updateNet();

    return C;
}

double Trainer::calculateCost(const Sample& data) const
{
    return cost_fun(net.forwardProp(data.first), data.second);
}

double Trainer::calculateCost(const std::vector<Sample>& data) const
{
    double C = 0.;
    for (const auto& d : data)
        C += calculateCost(d);

    return C / data.size();
}

double Trainer::calculateCostAndGradient(const Sample& data, std::vector<Eigen::MatrixXd>& g_w, std::vector<Eigen::VectorXd>& g_b)
{
    auto act = net.forwardProp(data.first);
    double C = cost_fun(act, data.second);
    auto dCda = deriv_cost(act, data.second);
    net.backwardProp(dCda);

    act = data.first;
    g_w.clear();
    g_b.clear();

    for (auto& layer : net.getLayers())
    {
        g_b.push_back(layer.getDelta());
        g_w.push_back(layer.getDelta() * act.transpose());
        act = layer.getActivation();
    }

    return C;
}

double Trainer::calculateRegularization(double strength) const
{
    double R = 0.;

    for (auto& layer : net.getLayers())
    {
        R += layer.getWeight().squaredNorm();
        R += layer.getBias().squaredNorm();
    }

    return strength * R / 2.;
}

void Trainer::addRegularizationGradient(double strength)
{
    auto W_it = grad_W.begin();
    auto b_it = grad_b.begin();
    auto l_it = net.getLayers().cbegin();

    while (W_it != grad_W.end() && b_it != grad_b.end() && l_it != net.getLayers().cend())
    {
        *W_it += l_it->getWeight() * strength;
        *b_it += l_it->getBias() * strength;
        ++W_it;
        ++b_it;
        ++l_it;
    }
}

void Trainer::updateNet()
{
    auto W_it = grad_W.begin();
    auto b_it = grad_b.begin();
    auto l_it = net.getLayers().begin();

    while (W_it != grad_W.end() && b_it != grad_b.end() && l_it != net.getLayers().end())
    {
        l_it->getWeight() += -stepsize * (*W_it);
        l_it->getBias() += -stepsize * (*b_it);

        ++W_it;
        ++b_it;
        ++l_it;
    }
}
