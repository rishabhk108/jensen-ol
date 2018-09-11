// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
#include "metrics.h"
namespace jensen {

  bool compare(const std::pair<float,int>&i, const std::pair<float,int>&j){
      return i.first > j.first;
  }

  double logloss(Vector& y, Vector& pred)
  {
      // Assumes the true label y is (-1,+1) and pred is a probability vector between 0 and 1. Pred stores P(y = 1|x)
      int n = y.size();
      double loss = 0;
      assert(y.size() == pred.size());
      for (int i = 0; i < n; i++)
      {
          if (y[i] == 1)
              loss = loss - log(pred[i]);
          else if (y[i] == -1)
              loss = loss - log(1 - pred[i]);
          else
              std::cout << "Invalid y[i]. Please check the input for log loss\n";
      }
      return loss/n;
  }

  double AUROC(Vector& y, Vector& pred)
  {
      int n = y.size();
      assert(y.size() == pred.size());
      std::vector<std::pair<float,int> > data;
      std::vector<float> TParray; //roc true positives
      std::vector<float> FParray; //roc false positive
      for (int i = 0; i < n; i++)
          data.push_back(std::make_pair(pred[i],y[i]));
      sort(data.begin(),data.end(),compare);
      int L = data.size();
      int P = 0;
      int N = 0;
      for (int j = 0; j < data.size(); j++)
      {
          if (data[j].second==1) {
              P++;
          }
          else {
              N++;
          }
      }
      double f_i;
      double label;
      //init FP TP counters
      double FP = 0;
      double TP = 0;
      double f_prev = -std::numeric_limits<double>::infinity();
      std::vector<std::pair<float,float> > R;
      //loop through all data
      for (int i = 0; i < L; i++) {
        f_i = data[i].first;
        label = data[i].second;
        if (f_i != f_prev) {
          // add points to roc curves
          TParray.push_back(TP/P);
          FParray.push_back(FP/N);
          f_prev = f_i;
        }
        if (data[i].second==1) {
          TP = TP + 1;
        }
        else {
          FP = FP + 1;
        }
      }
      TParray.push_back(TP/P);
      FParray.push_back(FP/N);
      int size = TParray.size();
      float q1,q2,p1,p2;
      q1 = FParray[0];
      q2 = TParray[0];
      float area = 0.0;
      for (int i=1;i < size;++i) {
          p1 = FParray[i];
          p2 = TParray[i];
          area += sqrt(pow( ((1-q1)+(1-p1))/2 * (q2-p2),2));
          q1=p1;
          q2=p2;
      }
      return area;
  }

  double accuracy(Vector& y, Vector& pred)
  {
      double totalscore = 0;
      assert(y.size() == pred.size());
      for (int i = 0; i < y.size(); i++)
      {
          if (y[i] == pred[i])
          {
              totalscore++;
          }
      }
      return totalscore/y.size();
  }
}
