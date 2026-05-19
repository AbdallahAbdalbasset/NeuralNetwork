#include "bits/stdc++.h"
using namespace std;

struct Node {
    double val = 1;
    double grad = 0;
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

class MSE : public Loss {
    double forward(double o, double t){
        return ((o - t) * (o - t)) / 2.0;
    }
    
    double backProp(double o, double t){
        return o - t;
    }
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

class NeuralNetwork{
public:
    Loss* loss;
    Activation* activation;
    vector<vector<Node>> layers;
    vector<vector<vector<Node>>> weights;
    double lr;
        
    NeuralNetwork(vector<int> layersSize, Loss* loss, Activation* activation, double lr) : loss(loss), activation(activation), lr(lr) {
        if(!layersSize.size() || !loss || !activation) {
            cout<<"Invalid Network"<<endl;
            return;
        }
        
        // Force Single Out Layer
        if(layersSize.back() != 1) layersSize.push_back(1);
        
        // Allocate Weights and Layers
        for(int i = 0;i+1<layersSize.size();i++){
            weights.push_back(vector<vector<Node>>(layersSize[i]+1, vector<Node>(layersSize[i+1])));
            layers.push_back(vector<Node>(layersSize[i]));
        }
        
        layers.push_back(vector<Node>(1));
    }
    
    double forward(vector<double> input){
        if(input.size() != layers[0].size()){
            cout<<"Invalid Input"<<endl;
            return 0.0;
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
        
        return layers.back()[0].val;
    }
    
    void backProp(double target){
        
        layers.back()[0].grad = loss->backProp(layers.back()[0].val, target);
        
        // Calc grad for each layer
        for(int i = (int)layers.size() - 2;i>=0;i--){
            for(int j = 0;j<layers[i].size();j++){
                layers[i][j].grad = 0;
                
                for(int k = 0;k<layers[i+1].size();k++){
                    layers[i][j].grad += layers[i+1][k].grad*weights[i][j][k].val;
                    weights[i][j][k].grad = layers[i][j].val*layers[i+1][k].grad;
                    weights[i][j][k].val -= lr * weights[i][j][k].grad;
                }
                
                layers[i][j].grad *= activation->backProp(layers[i][j].val);
            }
            
            // Bias grad
            for(int k = 0;k<layers[i+1].size();k++){
                weights[i][layers[i].size()][k].val -= lr * layers[i+1][k].grad;
            }
        }
    }
};



int main(){
    // Data
    vector<vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> Y = {0, 0, 0, 1};
    
    // Create Neural Network
    NeuralNetwork nn({2,1}, new MSE(), new Relu(), 0.01);
    
    // Training
    srand(1);
    for(int i = 0;i<40000;i++){
        int idx = rand()%4;
        nn.forward(X[idx]);
        nn.backProp(Y[idx]);
    }
    
    // Test
    for(int i = 0;i < 4;i++) {
        cout<<"Sample: "<<i+1<<endl;
        cout<<"Data: ("<<X[i][0]<<","<<X[i][1]<<")   Target "<<Y[i]<<"  Prediction: "<<nn.forward(X[i])<<endl;
    }
}
    



