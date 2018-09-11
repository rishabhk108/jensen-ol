// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
   Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Smooth SVM Loss with L2 regularization
   Author: Rishabh Iyer
 *
 */

#ifndef SMOOTHSVM_LOSS_H
#define SMOOTHSVM_LOSS_H

#include "../datarep/Vector.h"
#include "../datarep/Matrix.h"
#include "../datarep/VectorOperations.h"
#include "ContinuousFunctions.h"
namespace jensen {
template <class Feature>
class SmoothSVMLoss : public ContinuousFunctions {
protected:
std::vector<Feature>& features;     // size of features is number of trainins examples (n)
Vector& y;     // size of y is number of training examples (n)
public:
SmoothSVMLoss(int numFeatures, std::vector<Feature>& features, Vector& y);
SmoothSVMLoss(const SmoothSVMLoss& c);     // copy constructor

~SmoothSVMLoss();

double eval(const Vector& x) const;     // functionEval
Vector evalGradient(const Vector& x) const;     // gradientEval
void eval(const Vector& x, double& f, Vector& gradient) const;     // combined function and gradient eval
Vector evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const;     // stochastic gradient
void evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const;     // stochastic evaluation
};

}
#endif
