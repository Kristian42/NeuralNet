#include "activation_functions.h"
#include "cost_function.h"
#include "net.h"
#include "layer.h"
#include "trainer.h"
#include "mnistreader.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <utility>
#include <random>
#include <string>
#include <algorithm>


int main()
{
    using std::cout;
    using std::endl;
    using std::string;
    using std::vector;
    using namespace Eigen;


    // Setup the network
    const int num_input = MNISTReader::IMAGE_SIZE;
    const int num_hidden1 = 100;
    const int num_hidden2 = 50;
    const int num_output = 10;

    MatrixXd Wf = 2. * MatrixXd::Random(num_hidden1, num_input) / std::sqrt(num_input);

    VectorXd bf = 2. * VectorXd::Random(num_hidden1) / std::sqrt(num_input);

    MatrixXd Wm = 2. * MatrixXd::Random(num_hidden2, num_hidden1) / std::sqrt(num_hidden1);

    VectorXd bm = 2. * VectorXd::Random(num_hidden2) / std::sqrt(num_hidden1);

    MatrixXd Wl = 2. * MatrixXd::Random(num_output, num_hidden2) / std::sqrt(num_hidden2);

    VectorXd bl = 2. * VectorXd::Random(num_output) / std::sqrt(num_hidden2);

    Layer first(Wf, bf, lrelu, lrelu_deriv);
    Layer middle(Wm, bm, lrelu, lrelu_deriv);
    Layer last(Wl, bl, lrelu, lrelu_deriv);

    Net skynet;
    skynet.addLayer(first);
    skynet.addLayer(middle);
    skynet.addLayer(last);

    // Read the training data
    MNISTReader reader;
    string image_file = "/path/to/MNIST/testimages.bin";
    string label_file = "/path/to/MNIST/testlabels.bin";
    auto training_data = reader.readData(image_file, label_file);

    //Shuffle the training data
    shuffle(training_data.begin(), training_data.end(), std::default_random_engine(10));
    std::for_each(training_data.begin(), training_data.end(), [](Trainer::Sample& s){s.first /= 255.;});


    // Train the network
    double learning_rate = 0.5;
    constexpr double reg_strength = 1.e-3;
    Trainer coach(skynet, learning_rate, softmax_cost, softmax_cost_deriv);

    const size_t sample_size = 200;
    const auto num_batch = training_data.size() / sample_size - 1;   //Last batch reserved for validation
    const size_t num_iter = 25;

    for (size_t i = 0; i < num_iter; ++i)
    {
        for (size_t j = 0; j < num_batch; ++j)
        {
            double R = coach.calculateRegularization(reg_strength);
            double C = coach.train(training_data.begin() + j * sample_size, training_data.begin() + (j + 1) * sample_size, learning_rate, reg_strength);
            cout << "Iteration: " << i << " Batch: " << j << " Cost: " << C << " Regul: " << R << "\n";
        }
    }

    // Run validation
    std::vector<Trainer::Sample> test_data(training_data.end() -sample_size, training_data.end());
    cout << "Running validation...\n";

    int num_errors = 0;
    std::vector<std::pair<Trainer::Sample, Eigen::VectorXd>> errors;
    for (auto& d : test_data)
    {
        int index_pred = 0;
        int index_true = 0;
        Eigen::VectorXd y_pred = softmax(skynet.forwardProp(d.first));
        y_pred.maxCoeff(&index_pred);
        d.second.maxCoeff(&index_true);

        if (index_pred != index_true)
        {
            ++num_errors;
            errors.push_back(std::make_pair(d, y_pred));
        }
    }
    cout << "Number of test errors: " << num_errors << "\n";
    cout << "Test error ratio: " << float(num_errors) / test_data.size() << endl;
    cout << "Cost function for test data: " << coach.calculateCost(test_data) << endl;
    cout << "First 5 errors:\n";
    for (int i = 0; i < 5; ++i)
    {
        int index_pred;
        int index_true;
        errors[i].first.second.maxCoeff(&index_true);
        errors[i].second.maxCoeff(&index_pred);
        cout << "True: " << index_true << " Pred: " << index_pred << "\n";
    }

    return 0;
}
