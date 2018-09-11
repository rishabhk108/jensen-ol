// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        A test suite for classification (similar to lib-linear)
        Author: John Halloran
 *
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include "../src/jensen.h"
using namespace jensen;
using namespace std;
char* trainFile = NULL;
char* testFile = NULL;
int method = 5;
int nClasses = 2;
double lambda = 1;
char* outFile = NULL;
int maxIter = 250;
int algtype = 0;
double tau = 1e-4;
double eps = 1e-4;
int verb = 0;
char* help = NULL;

#define L1LR 1
#define L2LR 2
#define L1SSVM 3
#define L2SSVM 4
#define L2HSVM 5

// #define TEST

Arg Arg::Args[]={
	Arg("trainFile", Arg::Req, trainFile, "the input training data file",Arg::SINGLE),
	Arg("testFile", Arg::Req, testFile, "the input test data file",Arg::SINGLE),
	Arg("nClasses", Arg::Opt, nClasses, "The number of classes", Arg::SINGLE),
	Arg("method", Arg::Opt, method, "Training method: 1(L1LR), 2(L2LR), 3(L1SSVM), 4(L2SSVM), 5(L2HSVM)", Arg::SINGLE),
	Arg("reg", Arg::Opt, lambda, "Regularization parameter (default 1)", Arg::SINGLE),
	Arg("maxIter", Arg::Opt, maxIter, "Maximum number of iterations (default 250)", Arg::SINGLE),
	Arg("epsilon", Arg::Opt, eps, "epsilon for convergence (default: 1e-2)", Arg::SINGLE),
	Arg("algtype", Arg::Opt, algtype, "type of algorithm for training the corresponding method",Arg::SINGLE),
	Arg("model", Arg::Opt, outFile, "saving the training model",Arg::SINGLE),
	Arg("verb", Arg::Opt, verb, "verbosity",Arg::SINGLE),
	Arg("help", Arg::Help, help, "Print this message"),
	Arg()
};

string algs[] = {"L1 Logistic Regression", "L2 Logistic Regression", "L1 Smooth SVM",
	         "L2 Smooth SVM", "L2 Hinge SVM"};

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

int main(int argc, char** argv){
	bool parse_was_ok = Arg::parse(argc,(char**)argv);
	if(!parse_was_ok) {
		Arg::usage(); exit(-1);
	}

	int ntrain; // number of data items in the training set
	int mtrain; // numFeatures of the training data
	int ntest; // number of data items in the test set
	int mtest; // numFeatures of the test data
	vector<struct SparseFeature> trainFeatures, testFeatures;
	Vector ytrain, ytest;
	readFeatureLabelsLibSVM(trainFile, trainFeatures, ytrain, ntrain, mtrain);
	readFeatureLabelsLibSVM(testFile, testFeatures, ytest, ntest, mtest);
	cout << "Done reading the file, the size of the training set is " << ytrain.size() << " and the size of the test set is " <<ytest.size() << endl;
	cout << "Number of features of the train set is " << mtrain << " and the number of features of the test set is " << mtest << "\n";
	if ((method < 0) || (method > 5)) {
		cout << "Invalid method.\n";
		return -1;
	}
	cout << "Now training a " << algs[method-1] << " classifier.\n";
	cout << trainFeatures[1].featureVec[1] << " " << trainFeatures[1].featureIndex[1] << "\n";
	double accuracytest, accuracytrain = 0;
	double loglosstest, loglosstrain = 0;
	double auctest, auctrain = 0;
	Vector probtrain, probtest;
	if (method == L1LR) {
		Classifiers<SparseFeature>* c = new L1LogisticRegression<SparseFeature>(trainFeatures, ytrain, mtrain, ntrain, nClasses,
		                                                                        lambda, algtype, maxIter, eps);
		c->train();
		cout << "Done with Training ... now testing\n";
		accuracytest = predictAccuracy(c, testFeatures, ytest);
		probtest = predictProb(c, testFeatures);
		accuracytrain = predictAccuracy(c, trainFeatures, ytrain);
		probtrain = predictProb(c, trainFeatures);
		delete c;
	}
	else if (method == L2LR) {
		int pos = 0;
		int neg = 0;
		for(int i=0; i < ntrain; i++)
			if(ytrain[i] > 0)
				pos++;
		neg = ntrain - pos;
		eps = eps*max(min(pos,neg), 1)/ntrain;
		Classifiers<SparseFeature>* c = new L2LogisticRegression<SparseFeature>(trainFeatures, ytrain, mtrain, ntrain, nClasses,
		                                                                        lambda, algtype, maxIter, eps);
		c->train();
		cout << "Done with Training ... now testing\n";
		accuracytest = predictAccuracy(c, testFeatures, ytest);
		probtest = predictProb(c, testFeatures);
		accuracytrain = predictAccuracy(c, trainFeatures, ytrain);
		probtrain = predictProb(c, trainFeatures);
		delete c;

	}
	else if (method == L1SSVM) {
		Classifiers<SparseFeature>* c = new L1SmoothSVM<SparseFeature>(trainFeatures, ytrain, mtrain, ntrain, nClasses,
		                                                               lambda, algtype, maxIter, eps);
		c->train();
		cout << "Done with Training ... now testing\n";
		accuracytest = predictAccuracy(c, testFeatures, ytest);
		probtest = predictProb(c, testFeatures);
		accuracytrain = predictAccuracy(c, trainFeatures, ytrain);
		probtrain = predictProb(c, trainFeatures);
		delete c;
	}
	else if (method == L2SSVM) {
		Classifiers<SparseFeature>* c = new L2SmoothSVM<SparseFeature>(trainFeatures, ytrain, mtrain, ntrain, nClasses,
		                                                               lambda, algtype, maxIter, eps);
		c->train();
		cout << "Done with Training ... now testing\n";
		accuracytest = predictAccuracy(c, testFeatures, ytest);
		probtest = predictProb(c, testFeatures);
		accuracytrain = predictAccuracy(c, trainFeatures, ytrain);
		probtrain = predictProb(c, trainFeatures);
		delete c;
	}
	else if (method == L2HSVM) {
		Classifiers<SparseFeature>* c = new L2HingeSVM<SparseFeature>(trainFeatures, ytrain, mtrain, ntrain, nClasses,
		                                                              lambda, algtype, maxIter, eps);
		c->train();
		cout << "Done with Training ... now testing\n";
		accuracytest = predictAccuracy(c, testFeatures, ytest);
		probtest = predictProb(c, testFeatures);
		accuracytrain = predictAccuracy(c, trainFeatures, ytrain);
		probtrain = predictProb(c, trainFeatures);
		delete c;
	}
	else
	{
		cout << "Invalid mode\n";
		return -1;
	}
	std::cout << "Train Accuracy of the model is " << accuracytrain << "\n";
	std::cout << "Test Accuracy of the model is " << accuracytest << "\n";
	auctest = AUROC(ytest, probtest);
	auctrain = AUROC(ytrain, probtrain);
	std::cout << "Train AUCROC of the model is " << auctrain << "\n";
	std::cout << "Test AUCROC of the model is " << auctest << "\n";
	loglosstest = logloss(ytest, probtest);
	loglosstrain = logloss(ytrain, probtrain);
	std::cout << "Train LogLoss of the model is " << loglosstrain << "\n";
	std::cout << "Test LogLoss of the model is " << loglosstest << "\n";
}
