// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
 *	L2 Regularized Logistic Regression (with both two class and multi-class classification)
        Author: Rishabh Iyer

    algtype: type of algorithm:
                0 (TRON)
                1 (LBFGS)
                2 (Gradient Descent with Fixed Learning Rate),
                3 (Gradient Descent with Line Search),
                4 (Gradient Descent with Barzelie Borwein step size),
                5 (Conjugate Gradient),
                6 (Nesterov's optimal method),
                7 (Stochastic Gradient Descent with fixed step length)
                8 (Stochastic Gradient Descent with decaying step size)
                9 (Adaptive Gradient Algorithm (AdaGrad))
 *
 */

#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

using namespace std;

#include "BinaryL2LogisticRegression.h"
#include "../../optimization/contAlgorithms/contAlgorithms.h"
#include "../../optimization/contFunctions/L2LogisticLoss.h"
#include "../../representation/Set.h"


#define EPSILON 1e-6
namespace jensen {
template <class Feature>
BinaryL2LogisticRegression<Feature>::BinaryL2LogisticRegression(vector<Feature>& trainFeatures, Vector& y, int m, int n,
                                                    double lambda, int algtype, int update_algtype, int maxIter, int updateIter,
                                                    double LR, double updateLR, double eps, int miniBatch, int lbfgsMemory) :
                                                    Classifiers<Feature>(m, n), trainFeatures(trainFeatures), y(y),
                                                    lambda(lambda), algtype(algtype), update_algtype(update_algtype), maxIter(maxIter),
  updateIter(updateIter), LR(LR), updateLR(updateLR), eps(eps), miniBatch(miniBatch), lbfgsMemory(lbfgsMemory) {}

template <class Feature>
BinaryL2LogisticRegression<Feature>::BinaryL2LogisticRegression(const BinaryL2LogisticRegression<Feature>& c) : Classifiers<Feature>(m, n),
	trainFeatures(c.trainFeatures), y(c.y), lambda(c.lambda), algtype(c.algtype), update_algtype(c.update_algtype),
	maxIter(c.maxIter), updateIter(c.updateIter), LR(c.LR), updateLR(c.updateLR), eps(c.eps), miniBatch(c.miniBatch), lbfgsMemory(c.lbfgsMemory) {
}

template <class Feature>
BinaryL2LogisticRegression<Feature>::~BinaryL2LogisticRegression(){
}

template <class Feature>
void BinaryL2LogisticRegression<Feature>::train(){
  // train L2 regularized logistic regression
    Vector w0 = Vector (m, 0);
    minimize(trainFeatures, y, lambda, maxIter, algtype, w0, w, LR);
}

template <class Feature>
void BinaryL2LogisticRegression<Feature>::update(vector<Feature>& incTrainFeatures, Vector& incy){
  // Update L2 regularized logistic regression
    Vector wnew = w;
    minimize(incTrainFeatures, incy, lambda, updateIter, update_algtype, w, wnew, updateLR);
    w = wnew;
}

template <class Feature>
Vector BinaryL2LogisticRegression<Feature>::get_weights(){         // train L2 regularized logistic regression
    return w;
}

template <class Feature>
void BinaryL2LogisticRegression<Feature>::initialize_weights(Vector& wprime){         // train L2 regularized logistic regression
    w = wprime;
}

template <class Feature>
void BinaryL2LogisticRegression<Feature>::minimize(vector<Feature>& trainFeatures, Vector& labels, double lambda, int maxiter, int algotype, Vector& x0, Vector& x, double alpha)
{
  x = Vector(x0.size(), 0);
  L2LogisticLoss<Feature> ll(m, trainFeatures, labels, lambda);
  n = trainFeatures.size();
	if (algotype == 0) {
		cout<<"*******************************************************************\n";
		cout<<"Training using Trust Region Newton Algorithm...\n";
		x = tron(ll, x0, maxiter, eps);
	}
	else if (algotype == 1) {
		cout<<"*******************************************************************\n";
		cout<<"Training using LBFGS...\n";
		x = lbfgsMin(ll, x0, 1, alpha, maxiter, lbfgsMemory, eps);
	}
  else if (algotype == 2) {
  	cout<<"*******************************************************************\n";
  	cout<<"Training using Gradient Descent with fixed step size...\n";
  	x = gd(ll, x0, alpha, maxiter, eps);
  }
	else if (algotype == 3) {
		cout<<"*******************************************************************\n";
		cout<<"Training using Gradient Descent with Line Search...\n";
		x = gdLineSearch(ll, x0, 1, alpha, maxiter, eps);
	}
	else if (algotype == 4) {
		cout<<"*******************************************************************\n";
		cout<<"Training using Gradient Descent with Barzilia-Borwein Step Length\n";
		x = gdBarzilaiBorwein(ll, x0, 1, alpha, maxiter, eps);
	}
	// gradientDescentBB(ss, Vector(m, 0), 1, 1e-4, 250);
	else if (algotype == 5) {
		cout<<"*******************************************************************\n";
		cout<<"Training using Conjugate Gradient for Logistic Loss, press enter to continue...\n";
		x = cg(ll, x0, 1, alpha, maxiter, eps);
	}
	else if (algotype == 6) {
		cout<<"*******************************************************************\n";
		cout<<"Training using Nesterov's Method\n";
		x = gdNesterov(ll, x0, 1, alpha, maxiter, eps);
	}
	else if (algotype == 7) {
		cout<<"*******************************************************************\n";
		cout<<"Training using Stochastic Gradient Descent\n";
		x = sgd(ll, x0, n, alpha, miniBatch, eps, maxiter);
	}
	else if (algotype == 8) {
		cout<<"*******************************************************************\n";
		cout<<"Training using Stochastic Gradient Descent with decaying learning rate\n";
		x = sgdDecayingLearningRate(ll, x0, n, alpha, miniBatch, eps, maxiter);
	}
	else if (algotype == 9) {
		cout<<"*******************************************************************\n";
		cout<<"Training using Adaptive Gradient Algorithm\n";
		x = sgdAdagrad(ll, Vector(m, 0), n, alpha, miniBatch, eps, maxiter);
	}
  return;
}

// save the model
template <class Feature>
int BinaryL2LogisticRegression<Feature>::saveModel(char* model){
	FILE *fp = fopen(model,"w");
	if(fp==NULL) return -1;

	fprintf(fp, "algtype %d\n", algtype);
	fprintf(fp, "nFeatures %d\n", m);
	fprintf(fp, "n %d\n", n);
	fprintf(fp, "w\n");
		for(int i=0; i<w.size(); i++)
		{
			fprintf(fp, "%.16g ", w[i]);
		}
		fprintf(fp, "\n");
	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

//brief: Load an already saved model of the classifier
template <class Feature>
int BinaryL2LogisticRegression<Feature>::loadModel(char* model){
	FILE *fp = fopen(model,"r");
	if(fp==NULL) return -1;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"algtype")==0)
			fscanf(fp,"%d",&algtype);
		else if(strcmp(cmd,"nFeatures")==0)
			fscanf(fp,"%d",&m);
		else if(strcmp(cmd,"n")==0)
			fscanf(fp,"%d",&n);
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			return -1;
		}
	}
	w = Vector(m, 0);
	for (int i = 0; i < m; i++)
		  fscanf(fp, "%f ", w[i]);
	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	return 1;
}

template <class Feature>
double BinaryL2LogisticRegression<Feature>::predict(const Feature& testFeature, double& val){
	// the assumption here is that train and test datasets have the same number of features
		val = featureProductCheck(w, testFeature);
		double argval = 0;
		if (val > 0)
			argval = 1;
		else
			argval = -1;
		return argval;
}

template <class Feature>
double BinaryL2LogisticRegression<Feature>::predict(const Feature& testFeature){
	// the assumption here is that train and test datasets have the same number of features
	double val;
	val = featureProductCheck(w, testFeature);
	double argval = 0;
	if (val > 0)
		argval = 1;
	else
		argval = -1;
	return argval;
}

// prob is a vector. The assumption is that prob[0] corresponds to -1 and prob[1] corresponds to +1 in binary
// classification.
template <class Feature>
void BinaryL2LogisticRegression<Feature>::predictProbability(const Feature& testFeature, Vector& prob){
	// the assumption here is that train and test datasets have the same number of features
	prob = Vector(2, 0);
	double val;
	val = featureProductCheck(w, testFeature);
	prob[1] = 1/(1+exp(-val));
	prob[0] = 1 - prob[1];
	return;
}

template class BinaryL2LogisticRegression<SparseFeature>;
template class BinaryL2LogisticRegression<DenseFeature>;
}
