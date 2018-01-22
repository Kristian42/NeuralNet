#include "mnistreader.h"

#include <Eigen/Dense>

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>



MNISTReader::MNISTReader()
{
    //ctor
}

std::vector<unsigned char> MNISTReader::readLabels(const std::string& filename) const
{
    std::ifstream input(filename, std::ios::binary);
    if (!input.good())
        throw std::runtime_error("Error opening file: " + filename);

    input.seekg(LABEL_OFFSET);
    std::istreambuf_iterator<char> beg(input);
    std::istreambuf_iterator<char> eos;
    std::vector<unsigned char> labels(beg, eos);

    return labels;
}

std::vector<Eigen::VectorXd> MNISTReader::readImages(const std::string& filename) const
{
    std::ifstream input(filename, std::ios::binary);
    if (!input.good())
        throw std::runtime_error("Error opening file: " + filename);

    input.seekg(IMAGE_OFFSET);
    std::istreambuf_iterator<char> iter(input);
    std::istreambuf_iterator<char> eos;

    Eigen::VectorXd buffer(IMAGE_SIZE);
    std::vector<Eigen::VectorXd> images;

    while (iter != eos)
    {
        for (int i = 0; i < IMAGE_SIZE && iter != eos; ++i, ++iter)
        {
            buffer(i) = static_cast<unsigned char>(*iter);
        }

        images.push_back(buffer);
    }

    return images;
}

std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> MNISTReader::readData(const std::string& image_filename, const std::string& label_filename) const
{
    auto images = readImages(image_filename);
    auto labels = readLabels(label_filename);
    if (images.size() != labels.size())
        throw std::runtime_error("Inconsistent sizes of images and labels vectors.");

    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> data;
    data.reserve(images.size());

    for (size_t i = 0; i < images.size(); ++i)
    {
        Eigen::VectorXd label = Eigen::VectorXd::Zero(10);
        label(labels[i]) = 1.;
        data.push_back(std::make_pair(images[i], label));
    }

    return data;
}
