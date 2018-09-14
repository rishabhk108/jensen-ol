# Jensen-OL 
Jensen-OL is built on top of Jensen (available here: https://github.com/rishabhk108/jensen)

## Description
Jensen-OL implements the algorithms discussed in the following paper:

Rishabh Iyer, Nimit Acharya, Tanuja Bompada, Denis Charles, Eren Manavoglu, **A Unified Batch Online Learning Framework for Click Prediction**, Arxiv Preprint: arXiv:1809.04673 (https://arxiv.org/abs/1809.04673)

Jensen-OL includes support for Online Learning for Logistic Regression. Currently there are two algorithms implemented. One is the Early Stopping algorithm (ES) and another is Proximal Scheme (Prox). The ES algorithms and updates are implemented in `BinaryL2LogisticRegression` while the Prox one is in `ProxL2LogisticRegression`. You can initialize the models using `initialize_weights` function, while the models can be updated (incrementally) using `update` function.

Please see `onlinelearning/OLexample.cc` for an example of this.

## How to run Jensen-OL
You can run OLexample by invoking the following command line:
./OLexample -proxLambda 1 -esIter 1 -esAlgo 2

Currently the data directory and paths are in the executable hardcoded. We will provide these as command line arguments in the next release.

## ReadMe for Jensen

Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
Licensed under the Open Software License version 3.0
See COPYING or http://opensource.org/licenses/OSL-3.0

Please see https://github.com/rishabhk108/jensen for more information
