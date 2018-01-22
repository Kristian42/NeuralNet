#ifndef MNISTREADER_H
#define MNISTREADER_H

#include <Eigen/Dense>

#include <vector>
#include <string>
#include <utility>

class MNISTReader
{
    public:
        static constexpr int IMAGE_SIZE   = 28 * 28;

        MNISTReader();

        std::vector<unsigned char> readLabels(const std::string& filename) const;
        std::vector<Eigen::VectorXd> readImages(const std::string& filename) const;
        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> readData(const std::string& image_filename, const std::string& label_filename) const;


    private:

        static constexpr int LABEL_OFFSET = 8;
        static constexpr int IMAGE_OFFSET = 16;
};

#endif // MNISTREADER_H
