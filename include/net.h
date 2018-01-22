#ifndef NET_H
#define NET_H

#include "layer.h"

#include <Eigen/Dense>
#include <vector>


class Net
{
    public:

        Net();

        void addLayer(const Layer& layer);

        const std::vector<Layer>& getLayers() const;
        std::vector<Layer>& getLayers();

        Eigen::VectorXd forwardProp(const Eigen::VectorXd& input);
        Eigen::VectorXd backwardProp(const Eigen::VectorXd& error);

    private:
        std::vector<Layer> layers;
};

#endif // NET_H
