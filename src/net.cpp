#include "net.h"
#include "layer.h"

#include <stdexcept>

Net::Net() :
    layers()
{
    //ctor
}

void Net::addLayer(const Layer& layer)
{
    if (!layers.empty())
    {
        if (layers.back().getWeight().rows() != layer.getWeight().cols())
            throw std::runtime_error("Added layer not compatible with previous layer.");
    }
    layers.push_back(layer);
}

const std::vector<Layer>& Net::getLayers() const
{
    return layers;
}

std::vector<Layer>& Net::getLayers()
{
    return layers;
}

Eigen::VectorXd Net::forwardProp(const Eigen::VectorXd& input)
{
    Eigen::VectorXd a = input;
    for (auto& layer : layers)
    {
        a = layer.forwardProp(a);
    }

    return a;
}

Eigen::VectorXd Net::backwardProp(const Eigen::VectorXd& error)
{
    Eigen::VectorXd delta = error;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
    {
        delta = it->backwardProp(delta);
    }

    return delta;
}
