// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
#ifndef ML_METRICS
#define ML_METRICS

#include "../representation/Vector.h"
#include "../representation/Matrix.h"
#include "../representation/VectorOperations.h"
#include "../representation/MatrixOperations.h"
#include <vector>
#include <math.h>
#include <iostream>
namespace jensen {
  double logloss(Vector& y, Vector& pred);
  double AUROC(Vector& y, Vector& pred);
  double accuracy(Vector& y, Vector& pred);
}
#endif
