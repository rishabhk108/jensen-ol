// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Author: Rishabh Iyer
 *
 */

#include <iostream>
#include <math.h>
using namespace std;

#include "L2LogisticLossProx.h"
#include "../../representation/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-6
#define MAX 1e2
namespace jensen {

void UpdateHessianVectorProdProx(vector<SparseFeature>& features, Vector& Hxv, Vector& w, int n){
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < features[i].featureIndex.size(); j++) {
			Hxv[features[i].featureIndex[j]] += w[i]*features[i].featureVec[j];
		}
	}
}

void UpdateHessianVectorProdProx(vector<DenseFeature>& features, Vector& Hxv, Vector& w, int n){
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < features[i].featureVec.size(); j++) {
			Hxv[j] += w[i]*features[i].featureVec[j];
		}
	}
}

template <class Feature>
L2LogisticLossProx<Feature>::L2LogisticLossProx(int m, std::vector<Feature>& features, Vector& y, Vector& x0, double lambda) :
	ContinuousFunctions(true, m, features.size()), features(features), y(y), x0(x0), lambda(lambda)
{
	if (n > 0)
		assert(features[0].numFeatures == m);
	assert(features.size() == y.size());
}

template <class Feature>
L2LogisticLossProx<Feature>::L2LogisticLossProx(const L2LogisticLossProx& l) :
	ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), x0(l.x0), lambda(l.lambda) {
}

template <class Feature>
L2LogisticLossProx<Feature>::~L2LogisticLossProx(){
}

template <class Feature>
double L2LogisticLossProx<Feature>::eval(const Vector& x) const {
	assert(x.size() == m);
	double sum = 0.5*lambda*((x-x0)*(x-x0));
	for (int i = 0; i < n; i++) {
		double preval = y[i]*(x*features[i]);
		if (preval > MAX)
			continue;
		else if (preval < -1*MAX)
			sum += preval;
		else
			sum += log(1 + exp(-preval));
	}
	return sum;
}

template <class Feature>
Vector L2LogisticLossProx<Feature>::evalGradient(const Vector& x) const {
	assert(x.size() == m);
	Vector g = lambda*(x - x0);
	for (int i = 0; i < n; i++) {
		double preval = y[i]*(x*features[i]);
		if (preval > MAX)
			g -= y[i]*features[i];
		else if (preval < -1*MAX)
			continue;
		else
			g -= (y[i]/(1 + exp(-preval)))*features[i];
	}
	return g;
}

template <class Feature>
void L2LogisticLossProx<Feature>::eval(const Vector& x, double& f, Vector& g) const {
	assert(x.size() == m);
	g = lambda* (x - x0);
	f = 0.5*lambda*((x-x0)*(x-x0));
	double val;
	for (int i = 0; i < n; i++) {
		double preval = y[i]*(x*features[i]);
		if (preval > MAX) {
			g -= (y[i]/(1 + exp(preval)))*features[i];
		}
		else if (preval < -1*MAX) {
			g -= (y[i]/(1 + exp(preval)))*features[i];
			f-=preval;
		}
		else{
			f += log(1 + exp(-preval));
			g -= (y[i]/(1 + exp(preval)))*features[i];
		}
	}
	return;
}

template <class Feature>
void L2LogisticLossProx<Feature>::evalHessianVectorProduct(const Vector& x, const Vector& v, Vector& Hxv) const {  // evaluate a product between a hessian and a vector
	Vector w(n, 0);
	Hxv = Vector(m, 0);
	for (int i = 0; i < n; i++) {
		double val = y[i]*(x*features[i]);
		double z = 1/(1 + exp(-val));
		double D = z*(1 - z);
		w[i] = D*(v*features[i]);
	}
	UpdateHessianVectorProdProx(features, Hxv, w, n);
	Hxv += lambda*v;
}

template <class Feature>
Vector L2LogisticLossProx<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	Vector g = lambda*(x - x0);
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double preval = y[*it]*(x*features[*it]);
		if (preval > MAX)
			g -= y[*it]*features[*it];
		else if (preval < -1*MAX)
			continue;
		else
			g -= (y[*it]/(1 + exp(-preval)))*features[*it];
	}
	return g;
}

template <class Feature>
void L2LogisticLossProx<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
	assert(x.size() == m);
	g = lambda*(x - x0);
	f = 0.5*lambda*((x - x0)*(x - x0));
	double val;
	for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
		double preval = y[*it]*(x*features[*it]);
		if (preval > MAX) {
			g -= (y[*it]/(1 + exp(preval)))*features[*it];
		}
		else if (preval < -1*MAX) {
			g -= (y[*it]/(1 + exp(preval)))*features[*it];
			f-=preval;
		}
		else{
			f += log(1 + exp(-preval));
			g -= (y[*it]/(1 + exp(preval)))*features[*it];
		}
	}
	return;
}

template class L2LogisticLossProx<SparseFeature>;
template class L2LogisticLossProx<DenseFeature>;


}
