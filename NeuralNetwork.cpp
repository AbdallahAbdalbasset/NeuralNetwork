#include "bits/stdc++.h"
using namespace std;

struct Node {
    double val = 1;
    double grad = 0;
    double velocity = 0;
};

class Activation{
public:
    virtual double forward(double x) = 0;
    virtual double backProp(double x) = 0;
};

class Loss{
public:
    virtual double forward(double o, double t) = 0;
    virtual double backProp(double o, double t) = 0;
};

class Optimizer{
public:
    vector<vector<vector<Node>>>& weights;
    Optimizer(vector<vector<vector<Node>>>& weights) : weights(weights) {}
    virtual void step() = 0;
};

class Linear : public Activation{
    double forward(double x){
        return x;
    }
    
    double backProp(double x){
        return 1;
    }
};

class Relu : public Activation{
    double forward(double x) {
        return max(0.0, x);
    }
    
    double backProp(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
};

class MSE : public Loss {
    double forward(double o, double t){
        return ((o - t) * (o - t)) / 2.0;
    }
    
    double backProp(double o, double t){
        return o - t;
    }
};

class SGD : public Optimizer{
    double lr;
public:
    SGD(vector<vector<vector<Node>>>& weights, double lr) : Optimizer(weights), lr(lr){}
    void step(){
        for(auto&i:weights){
            for(auto&j:i){
                for(auto&k:j){
                    k.val -= lr * k.grad;
                }
            }
        }
    }
};

class MomentumSGD : public Optimizer{
    double mc;
    double lr;
public:
    MomentumSGD(vector<vector<vector<Node>>>& weights, double lr, double mc) : Optimizer(weights), lr(lr), mc(mc){
    }
    void step(){
        for(auto&i:weights){
            for(auto&j:i){
                for(auto&k:j){
                    k.velocity = mc*k.velocity - lr * k.grad;
                    k.val = k.val + k.velocity;
                }
            }
        }
    }
};

class NeuralNetwork{
public:
    Loss* loss;
    Activation* activation;
    vector<vector<Node>> layers;
    vector<vector<vector<Node>>> weights;
        
    NeuralNetwork(vector<int> layersSize, Loss* loss, Activation* activation) : loss(loss), activation(activation){
        if(!layersSize.size() || !loss || !activation) {
            cout<<"Invalid Network"<<endl;
            return;
        }
                
        // Allocate Weights and Layers
        for(int i = 0;i+1<layersSize.size();i++){
            weights.push_back(vector<vector<Node>>(layersSize[i]+1, vector<Node>(layersSize[i+1])));
            layers.push_back(vector<Node>(layersSize[i]));
        }
        layers.push_back(vector<Node>(layersSize.back()));
        
    }
    
    vector<double> forward(vector<double> input){
        if(input.size() != layers[0].size()){
            cout<<"Invalid Input"<<endl;
            return {};
        }
        
        // Initialize Input Layer
        for(int i = 0;i<input.size();i++) layers[0][i].val = input[i];
        
        // Calc Forward Path For Each Layer
        for(int i = 1;i<layers.size();i++){
            for(int j = 0;j<layers[i].size();j++){
                layers[i][j].val = 0;
                
                for(int k = 0;k<layers[i-1].size();k++){
                    layers[i][j].val += layers[i-1][k].val*weights[i-1][k][j].val;
                }
                
                // Add bias
                layers[i][j].val += weights[i-1][layers[i-1].size()][j].val;
                layers[i][j].val = activation->forward(layers[i][j].val);
            }
        }
        
        // Return vector of outputs
        vector<double> res(layers.back().size());
        for(int i = 0;i<res.size();i++) res[i] = layers.back()[i].val;
        return res;
    }
    
    void backProp(vector<double> target){
        if(target.size() != layers.back().size()){
            cout<<"Target size does not mach the NN output"<<endl;
            return;
        }
        
        for(int i = 0;i<layers.back().size();i++){
            layers.back()[i].grad = loss->backProp(layers.back()[i].val, target[i])/layers.back().size();
        }
        
        // Calc grad for each layer
        for(int i = (int)layers.size() - 2;i>=0;i--){
            for(int j = 0;j<layers[i].size();j++){
                layers[i][j].grad = 0;
                
                for(int k = 0;k<layers[i+1].size();k++){
                    layers[i][j].grad += layers[i+1][k].grad*weights[i][j][k].val;
                    weights[i][j][k].grad = layers[i][j].val*layers[i+1][k].grad;
                }
                
                layers[i][j].grad *= activation->backProp(layers[i][j].val);
            }
            
            // Bias grad
            for(int k = 0;k<layers[i+1].size();k++){
                weights[i][layers[i].size()][k].grad = layers[i+1][k].grad;
            }
        }
    }
};



int main(){
    // Data
    vector<vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> Y = {{0,0}, {0,1}, {0,2}, {1,3}};
    
    // Create Neural Network
    NeuralNetwork nn({2,2}, new MSE(), new Relu());
    MomentumSGD msgd(nn.weights, 0.01, 0.9);
    // Training
    srand(1);
    for(int i = 0;i<40000;i++){
        int idx = rand()%4;
        nn.forward(X[idx]);
        nn.backProp(Y[idx]);
        msgd.step();
    }
    
    // Test
    for(int i = 0;i < 4;i++) {
        cout<<"Sample: "<<i+1<<endl;
        auto res = nn.forward(X[i]);
        cout<<"Data: ("<<X[i][0]<<","<<X[i][1]<<") Prediction: "<<res[0]<<" "<<res[1]<<endl;
    }
}
    



