// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        An Example for running Online Learning with Jensen
 *
 */

#include <iostream>
#include <cstdlib>
#include <sstream>  // for string streams
#include <string>  // for string
#include "../src/jensen.h"
using namespace jensen;
using namespace std;

double lambda = 0;
int maxIter = 50;
double LR = 1e-05;
int algtype = 0; // algtype of the base model
double tau = 1e-4;
double eps = 1e-2; // convergence criteria
int esIter = 5; // number of iterations of early stopping
int esAlgo = 7; // es update algo: SGD with fixed learning rate
double esLR = 1e-05; // learning rate of the early stopping
int proxAlgo = 1; // prox update algo: LBFGS
double proxLambda = 100;
int verb = 0;
char* help = NULL;
double epsilon = 1e-10;

Arg Arg::Args[]={
	Arg("reg", Arg::Opt, lambda, "Regularization parameter of base model: default 1", Arg::SINGLE),
	Arg("esIter", Arg::Opt, esIter, "Number of Iterations to run early stopping algo", Arg::SINGLE),
	Arg("esAlgo", Arg::Opt, esAlgo, "Early stopping update algo", Arg::SINGLE),
	Arg("esLR", Arg::Opt, esLR, "Early stopping update LR", Arg::SINGLE),
	Arg("proxAlgo", Arg::Opt, proxAlgo, "Proximal update Algo", Arg::SINGLE),
	Arg("proxLambda", Arg::Opt, proxLambda, "Proximal Lambda", Arg::SINGLE),
	Arg("epsilon", Arg::Opt, eps, "epsilon for convergence (default: 1e-4)", Arg::SINGLE),
	Arg("algtype", Arg::Opt, algtype, "algo type of the base model",Arg::SINGLE),
	Arg("verb", Arg::Opt, verb, "verbosity",Arg::SINGLE),
	Arg("help", Arg::Help, help, "Print this message"),
	Arg()
};

template <class Feature>
double predictAccuracy(Classifiers<Feature>* c, vector<Feature>& testFeatures, Vector& ytest){
	assert(testFeatures.size() == ytest.size());
	double accuracy = 0;
	for (int i = 0; i < testFeatures.size(); i++) {
		if (c->predict(testFeatures[i]) == ytest[i])
			accuracy++;
	}
	return accuracy/ytest.size();
}

template <class Feature>
Vector predictProb(Classifiers<Feature>* c, vector<Feature>& testFeatures){
	Vector prob(testFeatures.size(),0);
	Vector currprob;
	for (int i = 0; i < testFeatures.size(); i++) {
			c->predictProbability(testFeatures[i], currprob);
			prob[i] = currprob[1];
	}
	return prob;
}

template <class Feature>
void printMetrics(Classifiers<Feature>* c, vector<Feature>& features, Vector& y, std::string desc, double & accuracy, double & logl, double & auc,
	 double& RIG, bool print)
{
		Vector prob;
		accuracy = predictAccuracy(c, features, y);
		prob = predictProb(c, features);
		if (print)
			std::cout << "Accuracy of " << desc << " " << accuracy << "\n";
		auc = AUROC(y, prob);
		if (print)
			std::cout << "AUCROC of " << desc << " " << auc << "\n";
		logl = logloss(y, prob);
		if (print)
			std::cout << "LogLoss of " << desc << " " << logl << "\n";
		double CTR = 0;
		for (int i = 0; i < y.size(); i++)
		{
				if (y[i] == 1)
						CTR++;
		}
		CTR = CTR/y.size();
		double CTRval = -CTR*log(CTR) - (1-CTR)*log(1-CTR);
		RIG = 1 - logl/CTRval;
		if (print)
			std::cout << "RIG of " << desc << " " << RIG << "\n";

}

int main(int argc, char** argv)
{
	bool parse_was_ok = Arg::parse(argc,(char**)argv);
	if(!parse_was_ok) {
		Arg::usage(); exit(-1);
	}

	int nbase; // number of data items in the training set
	int mbase; // numFeatures of the training data
	vector<struct SparseFeature> baseFeatures;
	double accuracybase, aucbase, loglossbase, rigbase;
	Vector ybase;
	char* baseFile = "/Users/rkiyer/Downloads/OLMobileData/BaseData_LibSVM.tsv";
	readFeatureLabelsLibSVM(baseFile, baseFeatures, ybase, nbase, mbase);
	L2LogisticRegression<SparseFeature> lrbase(baseFeatures, ybase, mbase, nbase, 2, lambda,
	                     algtype, maxIter, eps);
	lrbase.train();
	Vector wbase = lrbase.get_weights();
	printMetrics(&lrbase, baseFeatures, ybase, "LRbase on base data ", accuracybase, loglossbase, aucbase, rigbase, false);
	BinaryL2LogisticRegression<SparseFeature> es(baseFeatures, ybase, mbase, nbase, lambda,
	                     algtype, esAlgo, maxIter, esIter, LR, esLR, eps);
	// es.train();
	es.initialize_weights(wbase);
	printMetrics(&es, baseFeatures, ybase, "ES on base data ", accuracybase, loglossbase, aucbase, rigbase, false);
	ProxL2LogisticRegression<SparseFeature> prox(baseFeatures, ybase, mbase, nbase, lambda, proxLambda,
	                     algtype, proxAlgo, maxIter, LR, eps);
	// prox.train();
	prox.initialize_weights(wbase);
	printMetrics(&prox, baseFeatures, ybase, "Prox on base data ", accuracybase, loglossbase, aucbase, rigbase, false);
	int numdays = 11;
	std::vector<double> aucES = std::vector<double>(numdays);
	std::vector<double> aucProx = std::vector<double>(numdays);
	std::vector<double> aucBase = std::vector<double>(numdays);
	std::vector<double> loglossES = std::vector<double>(numdays);
	std::vector<double> loglossProx = std::vector<double>(numdays);
	std::vector<double> loglossBase = std::vector<double>(numdays);
	std::vector<double> RIGES = std::vector<double>(numdays);
	std::vector<double> RIGProx = std::vector<double>(numdays);
	std::vector<double> RIGBase = std::vector<double>(numdays);
	// std::vector<double> lossDiff = std::vector<double>(numdays);
	// std::vector<double> maxGradDiff = std::vector<double>(numdays);

	double accuracy;
	for (int i = 1; i <= numdays; i++)
	{
			ostringstream num;
			num << i;
			string currFile = "/Users/rkiyer/Downloads/OLMobileData/Day" + num.str() + "_LibSVM.tsv";
			vector<struct SparseFeature> currFeatures;
			Vector ycurr;
			int ncurr, mcurr;
			// Vector w0_prox = prox.get_weights();
			// Vector w0_es = es.get_weights();
			readFeatureLabelsLibSVM(currFile.c_str(), currFeatures, ycurr, ncurr, mcurr);
			// L2LogisticLossProx<SparseFeature> p(mcurr, currFeatures, ycurr, w0, proxLambda);
			// L2LogisticLoss<SparseFeature> l(mcurr, currFeatures, ycurr, lambda);
			printMetrics(&lrbase, currFeatures, ycurr, "LRbase on data " + num.str() + " ", accuracy, loglossBase[i-1], aucBase[i-1], RIGBase[i-1], false);
			printMetrics(&prox, currFeatures, ycurr, "Prox on data " + num.str() + " ", accuracy, loglossProx[i-1], aucProx[i-1], RIGProx[i-1], false);
			printMetrics(&es, currFeatures, ycurr, "ES on data " + num.str() + " ", accuracy, loglossES[i-1], aucES[i-1], RIGES[i-1], false);
			es.update(currFeatures, ycurr);
			prox.update(currFeatures, ycurr);
			// Vector w_es = es.get_weights();
			// Vector w_prox = prox.get_weights();
			// lossDiff[i-1] = abs(p.eval(w_es) - p.eval(w_prox))/ncurr;
			// gd_maxDiff(l, w0_es, maxGradDiff[i], esLR, esIter);
			// maxGradDiff[i] = maxGradDiff[i]/ncurr;
			// std::cout << "Max Grad Diff of the current iteration is " << maxGradDiff[i] << "\n";
	}
	std::cout << "Results for esIter = " << esIter << " and proxLambda = " << proxLambda << " and esLR = " << esLR << " and esAlgo = " << esAlgo << "\n";
	std::cout << "Base AUC:\n";
	for (int i = 0; i < numdays; i++)
	{
			std::cout << aucBase[i] << "\n";
	}
	std::cout << "Prox AUC:\n";
	for (int i = 0; i < numdays; i++)
	{
			std::cout << aucProx[i] << "\n";
	}
	std::cout << "ES AUC:\n";
	for (int i = 0; i < numdays; i++)
	{
			std::cout << aucES[i] << "\n";
	}
	std::cout << "Base RIG:\n";
	for (int i = 0; i < numdays; i++)
	{
			std::cout << RIGBase[i] << "\n";
	}
	std::cout << "Prox RIG:\n";
	for (int i = 0; i < numdays; i++)
	{
			std::cout << RIGProx[i] << "\n";
	}
	std::cout << "ES RIG:\n";
	for (int i = 0; i < numdays; i++)
	{
			std::cout << RIGES[i] << "\n";
	}
	/*std::cout << "LOSS DIFF PROX:\n";
	for (int i = 0; i < numdays; i++)
	{
			std::cout << abs(lossDiff[i]) << "\n";
	}
	std::cout << "MAX GRAD DIFF:\n";
	for (int i = 0; i < numdays; i++)
	{
			std::cout << (esIter - 1)*abs(maxGradDiff[i]) << "\n";
	}*/

}
