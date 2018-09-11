// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	L1 Regularized Logistic Regression (Useful if you want to encourage sparsity in the classifier)
        Author: Rishabh Iyer

    algtype: type of algorithm:
                0 (TRON)
                1 (LBFGS)
                2 (Gradient Descent),
                3 (Gradient Descent with Line Search),
                4 (Gradient Descent with Barzelie Borwein step size),
                5 (Conjugate Gradient),
                6 (Nesterov's optimal method),
                7 (Stochastic Gradient Descent with fixed step length)
                8 (Stochastic Gradient Descent with decaying step size)
                9 (Adaptive Gradient Algorithm (AdaGrad))
 *
 */

#ifndef BINARY_L2_LOGISTIC_REGRESSION_H
#define BINARY_L2_LOGISTIC_REGRESSION_H

#include "../Classifiers.h"
#include "../../representation/Vector.h"
#include "../../representation/Matrix.h"
#include "../../representation/VectorOperations.h"
#include "../../representation/MatrixOperations.h"
#include <vector>
using namespace std;

namespace jensen {

template <class Feature>
class BinaryL2LogisticRegression : public Classifiers<Feature>{
protected:
vector<Feature>& trainFeatures;                 // training features
Vector& y;             // size of y is number of training examples (n)
int algtype;                 // the algorithm type used for training, default is the trust region newton.
int update_algtype;
Vector w;                 // the weights in the binary scenario.
double lambda;                 // regularization
int maxIter;              // maximum number of iterations for the algorithms
int updateIter;
double LR;
double updateLR;
double eps;                 // stopping criterion for the algorithms
int miniBatch;                 // This is the miniBatch size for Stochastic Gradient Descent
int lbfgsMemory;                 // memory of the LBFGS algorithm.
using Classifiers<Feature>::m;
using Classifiers<Feature>::n;
public:
BinaryL2LogisticRegression(vector<Feature>& trainFeatures, Vector& y, int m, int n, double lambda = 1,
                     int algtype = 0, int update_algtype = 8, int maxIter = 250, int updateIter = 5,
                     double LR = 1e-05, double updateLR = 1e-05,
                     double eps = 1e-4, int miniBatch = 100, int lbfgsMemory = 100);
BinaryL2LogisticRegression(const BinaryL2LogisticRegression& c);         // copy constructor
~BinaryL2LogisticRegression();

void train();                 // train
void update(vector<Feature>& incTrainFeatures, Vector& incy);
int saveModel(char* model);                 // save the model
int loadModel(char* model);                 // save the model

double predict(const Feature& testFeature);
double predict(const Feature& testFeature, double& val);
void predictProbability(const Feature& testFeature, Vector& prob);
Vector get_weights();
void initialize_weights(Vector& w);
private:
  void minimize(vector<Feature>& trainFeatures, Vector& labels, double lambda, int maxiter, int algotype, Vector& x0, Vector& x, double alpha = 1e-04);
};

}
#endif
