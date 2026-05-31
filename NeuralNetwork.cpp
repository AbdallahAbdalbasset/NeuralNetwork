#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <random>

using namespace std;

struct Weight
{   // Split weight and Neuron and store pre and post activation, backward of Relu takes the input not the out of the net
    double val = 1;
    double grad = 0;
    double velocity = 0;
};

struct Neuron{
    double grad = 0;
    double netVal = 0;
    double outVal = 0;
};

double randomWeight(){
    return ((double)rand() / RAND_MAX - 0.5) * 0.1;
}

class Activation
{
public:
    virtual double forward(double x) = 0;
    virtual double backProp(double x) = 0;
};

class Loss
{
public:
    virtual double forward(double o, double t) = 0;
    virtual double backProp(double o, double t) = 0;
};

class Optimizer
{
public:
    vector<vector<vector<Weight>>> &weights;
    Optimizer(vector<vector<vector<Weight>>> &weights) : weights(weights) {}
    virtual void step() = 0;
};

class Linear : public Activation
{
public:
    double forward(double x)
    {
        return x;
    }

    double backProp(double x)
    {
        return 1;
    }
};

class Relu : public Activation
{
public:
    double forward(double x)
    {
        return max(0.0, x);
    }

    double backProp(double x)
    {
        return x > 0 ? 1.0 : 0.0;
    }
};

class MSE : public Loss
{
public:
    double forward(double o, double t)
    {
        return ((o - t) * (o - t)) / 2.0;
    }

    double backProp(double o, double t)
    {
        return o - t;
    }
};

class SGD : public Optimizer
{
    double lr;

public:
    SGD(vector<vector<vector<Weight>>> &weights, double lr) : Optimizer(weights), lr(lr) {}
    void step()
    {
        for (auto &i : weights)
        {
            for (auto &j : i)
            {
                for (auto &k : j)
                {
                    k.val -= lr * k.grad;
                }
            }
        }
    }
};

class MomentumSGD : public Optimizer
{
    double mc;
    double lr;

public:
    MomentumSGD(vector<vector<vector<Weight>>> &weights, double lr, double mc) : Optimizer(weights), lr(lr), mc(mc){}
    void step()
    {
        for (auto &i : weights)
        {
            for (auto &j : i)
            {
                for (auto &k : j)
                {
                    k.velocity = mc * k.velocity - lr * k.grad;
                    k.val = k.val + k.velocity;
                }
            }
        }
    }
};

class NeuralNetwork
{
public:
    Loss *loss;
    Activation *activation;
    vector<vector<Neuron>> layers;
    vector<vector<vector<Weight>>> weights;

    NeuralNetwork(vector<int> layersSize, Loss *loss, Activation *activation) : loss(loss), activation(activation)
    {
        if (!layersSize.size() || !loss || !activation){
            cout << "Invalid Network" << endl;
            return;
        }

        srand(time(nullptr));
        // Allocate Weights and Layers
        for (int i = 0; i + 1 < layersSize.size(); i++){
            weights.push_back(vector<vector<Weight>>(layersSize[i] + 1, vector<Weight>(layersSize[i + 1])));

            // Initialize weights randomly
            for (int from = 0; from < layersSize[i]; from++){
                for (int to = 0; to < layersSize[i + 1]; to++){
                    weights.back()[from][to].val = randomWeight();
                }
            }

            // Initialize biases to zero
            for (int to = 0; to < layersSize[i + 1]; to++){
                weights.back()[layersSize[i]][to].val = 0.0;
            }
            
            layers.push_back(vector<Neuron>(layersSize[i]));
        }

        layers.push_back(vector<Neuron>(layersSize.back()));
    }

    vector<double> forward(vector<double> input)
    {
        if (input.size() != layers[0].size())
        {
            cout << "Invalid Input" << endl;
            return {};
        }

        // Initialize Input Layer
        for (int i = 0; i < input.size(); i++)
            layers[0][i].outVal = input[i];

        // Calc Forward Path For Each Layer
        for (int i = 1; i < layers.size(); i++)
        {
            for (int j = 0; j < layers[i].size(); j++)
            {
                layers[i][j].netVal = 0;

                for (int k = 0; k < layers[i - 1].size(); k++)
                {
                    layers[i][j].netVal += layers[i - 1][k].outVal * weights[i - 1][k][j].val;
                }

                // Add bias
                layers[i][j].netVal += weights[i - 1][layers[i - 1].size()][j].val;
                layers[i][j].outVal = activation->forward(layers[i][j].netVal);
            }
        }

        // Return vector of outputs
        vector<double> res(layers.back().size());
        for (int i = 0; i < res.size(); i++)
            res[i] = layers.back()[i].outVal;
        return res;
    }

    void backProp(vector<double> target)
    {
        if (target.size() != layers.back().size())
        {
            cout << "Target size does not mach the NN output" << endl;
            return;
        }

        for (int i = 0; i < layers.back().size(); i++)
        {
            layers.back()[i].grad = loss->backProp(layers.back()[i].outVal, target[i]) / layers.back().size();
            layers.back()[i].grad *= activation->backProp(layers.back()[i].netVal);
        }

        // Calc grad for each layer
        for (int i = (int)layers.size() - 2; i >= 0; i--)
        {
            for (int j = 0; j < layers[i].size(); j++)
            {
                layers[i][j].grad = 0;

                for (int k = 0; k < layers[i + 1].size(); k++)
                {
                    layers[i][j].grad += layers[i + 1][k].grad * weights[i][j][k].val;
                    weights[i][j][k].grad = layers[i][j].outVal * layers[i + 1][k].grad;
                }
                if(i > 0) layers[i][j].grad *= activation->backProp(layers[i][j].netVal);
            }

            // Bias grad
            for (int k = 0; k < layers[i + 1].size(); k++)
            {
                weights[i][layers[i].size()][k].grad = layers[i + 1][k].grad;
            }
        }
    }

    void saveNNtoFile(string fileName){
        // Open file
        std::ofstream file(fileName);

        // Network architecture
        file << layers.size() << '\n';
        for(auto&i:layers) file << i.size() << ' ';
        file << '\n';

        // Weights values
        for(auto&i:weights) for(auto&j:i) for(auto&k:j) file<<k.val<<'\n';

        file.close();
    }
};

struct MNISTDataset {
    vector<vector<double>> images;
    vector<int> labels;
};

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24)
         + ((int)c2 << 16)
         + ((int)c3 << 8)
         + c4;
}

MNISTDataset loadMNIST(
    const string& imageFile,
    const string& labelFile
) {
    MNISTDataset dataset;

    ifstream images(imageFile, ios::binary);
    ifstream labels(labelFile, ios::binary);

    if (!images.is_open() || !labels.is_open()) {
        cout << "Cannot open files\n";
        return dataset;
    }

    int magicNumber = 0;
    int numberOfImages = 0;
    int rows = 0;
    int cols = 0;

    // IMAGE HEADER
    images.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);

    images.read((char*)&numberOfImages, sizeof(numberOfImages));
    numberOfImages = reverseInt(numberOfImages);

    images.read((char*)&rows, sizeof(rows));
    rows = reverseInt(rows);

    images.read((char*)&cols, sizeof(cols));
    cols = reverseInt(cols);

    // LABEL HEADER
    int labelMagic = 0;
    int numberOfLabels = 0;

    labels.read((char*)&labelMagic, sizeof(labelMagic));
    labelMagic = reverseInt(labelMagic);

    labels.read((char*)&numberOfLabels, sizeof(numberOfLabels));
    numberOfLabels = reverseInt(numberOfLabels);

    // READ DATA
    for (int i = 0; i < numberOfImages; i++) {

        vector<double> image(rows * cols);

        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel = 0;
            images.read((char*)&pixel, sizeof(pixel));

            // normalize to 0..1
            image[j] = pixel / 255.0;
        }

        unsigned char label = 0;
        labels.read((char*)&label, sizeof(label));

        dataset.images.push_back(image);
        dataset.labels.push_back((int)label);
    }

    return dataset;
}

void savePGM(const std::vector<double>& image,
             const std::string& filename)
{
    std::ofstream out(filename);

    out << "P2\n";
    out << "28 28\n";
    out << "255\n";

    for (int i = 0; i < 784; i++) {
        int pixel = static_cast<int>(image[i] * 255.0);
        out << pixel << ' ';

        if ((i + 1) % 28 == 0)
            out << '\n';
    }
}

int main()
{
    MNISTDataset trainData = loadMNIST(
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte"
    );

    MNISTDataset testData = loadMNIST(
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte"
    );

    NeuralNetwork nn({28*28, 10}, new MSE(), new Relu());
    MomentumSGD sgd(nn.weights, 0.001, 0.9);

    std::mt19937 rng(std::random_device{}());
    for(int epoch = 0; epoch < 10;epoch++){
        vector<int> idx(trainData.images.size());
        iota(idx.begin(), idx.end(), 0);
        shuffle(idx.begin(), idx.end(), rng);

        for(int i = 0;i < trainData.images.size();i++){
            nn.forward(trainData.images[idx[i]]);

            vector<double> label(10, 0);
            label[trainData.labels[idx[i]]] = 1;

            nn.backProp(label);
            sgd.step();
        }

        // Track progress
        cout<<"Finished Epoch: "<<epoch + 1<<endl;
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::cout << std::ctime(&now_time);
    }

    // Testing
    int correct = 0;
    for(int j = 0;j<testData.images.size();j++){
        auto res = nn.forward(testData.images[j]);      
        auto idx = max_element(res.begin(), res.end()) - res.begin();
        correct += idx == testData.labels[j];
    }
    cout<<"Accuracy: " <<100.0*correct/testData.images.size()<<endl;

    // Save the NN to file
    nn.saveNNtoFile("Weights.txt");
}
