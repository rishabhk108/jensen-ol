// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	L2 Regularized Proximal Logistic Regression
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

#ifndef PROX_L2_LOGISTIC_REGRESSION_H
#define PROX_L2_LOGISTIC_REGRESSION_H

#include "../Classifiers.h"
#include "../../representation/Vector.h"
#include "../../representation/Matrix.h"
#include "../../representation/VectorOperations.h"
#include "../../representation/MatrixOperations.h"
#include <vector>
using namespace std;

namespace jensen {

template <class Feature>
class ProxL2LogisticRegression : public Classifiers<Feature>{
protected:
vector<Feature>& trainFeatures;                 // training features
Vector& y;             // size of y is number of training examples (n)
int algtype;                 // the algorithm type used for training, default is the trust region newton.
int update_algtype;
Vector w;                 // the weights in the binary scenario.
double lambda;                 // regularization
double updatelambda;
int maxIter;              // maximum number of iterations for the algorithms
double LR;
double eps;                 // stopping criterion for the algorithms
int miniBatch;                 // This is the miniBatch size for Stochastic Gradient Descent
int lbfgsMemory;                 // memory of the LBFGS algorithm.
using Classifiers<Feature>::m;
using Classifiers<Feature>::n;
public:
ProxL2LogisticRegression(vector<Feature>& trainFeatures, Vector& y, int m, int n, double lambda = 1, double updatelambda = 100,
                     int algtype = 0, int update_algtype = 8, int maxIter = 250, double LR = 1e-05,
                     double eps = 1e-4, int miniBatch = 100, int lbfgsMemory = 100);
ProxL2LogisticRegression(const ProxL2LogisticRegression& c);         // copy constructor
~ProxL2LogisticRegression();

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
